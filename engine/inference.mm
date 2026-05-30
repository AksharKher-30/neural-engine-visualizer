#import "inference.hpp"
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#include <chrono>

// ── Objective-C class — owns the MLModel, talks to CoreML ──


@interface ObjCEngine : NSObject
@property (strong) MLModel* model;
- (instancetype)initWithPath:(NSString*)path;
- (NSDictionary<NSString*, MLMultiArray*>*)runWithRGB:(const uint8_t*)rgb
                                                width:(int)w
                                               height:(int)h;
@end

@implementation ObjCEngine

- (instancetype)initWithPath:(NSString*)path {
    self = [super init];
    if (self) {
        NSError* err = nil;
        NSURL* url = [NSURL fileURLWithPath:path];

        // Step 1: compile .mlpackage → .mlmodelc (required for ObjC CoreML API)
        NSLog(@"[Engine] Compiling model (first run ~20s)...");
        NSURL* compiledURL = [MLModel compileModelAtURL:url error:&err];
        if (!compiledURL || err) {
            NSLog(@"[Engine] Compile FAILED: %@", err.localizedDescription);
            return self;
        }
        NSLog(@"[Engine] Compiled OK.");

        // Step 2: load compiled model
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = MLComputeUnitsAll;
        _model = [MLModel modelWithContentsOfURL:compiledURL
                                   configuration:cfg
                                           error:&err];
        if (!_model)
            NSLog(@"[Engine] Load FAILED: %@", err.localizedDescription);
    }
    return self;
}

- (NSDictionary<NSString*, MLMultiArray*>*)runWithRGB:(const uint8_t*)rgb
                                                width:(int)w
                                               height:(int)h {
    // 1. Build CVPixelBuffer (BGRA — required by CoreML ImageType on macOS)
    NSDictionary* opts = @{
        (id)kCVPixelBufferCGImageCompatibilityKey:        @YES,
        (id)kCVPixelBufferCGBitmapContextCompatibilityKey:@YES
    };
    CVPixelBufferRef pb = NULL;
    CVReturn ret = CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                                       kCVPixelFormatType_32BGRA,
                                       (__bridge CFDictionaryRef)opts, &pb);
    if (ret != kCVReturnSuccess) return nil;

    CVPixelBufferLockBaseAddress(pb, 0);
    uint8_t* dst     = (uint8_t*)CVPixelBufferGetBaseAddress(pb);
    size_t rowBytes  = CVPixelBufferGetBytesPerRow(pb);

    for (int y = 0; y < h; y++) {
        uint8_t* row = dst + y * rowBytes;
        for (int x = 0; x < w; x++) {
            int s = (y * w + x) * 3;   // RGB source
            int d = x * 4;              // BGRA dest
            row[d+0] = rgb[s+2];        // B ← R
            row[d+1] = rgb[s+1];        // G ← G
            row[d+2] = rgb[s+0];        // R ← B
            row[d+3] = 255;             // A
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, 0);

    // 2. Wrap in MLFeatureValue and build provider
    NSError* err = nil;
    MLFeatureValue* fv = [MLFeatureValue featureValueWithPixelBuffer:pb];

    MLDictionaryFeatureProvider* provider =
        [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{@"input": fv} error:&err];
    if (err) { NSLog(@"Provider error: %@", err); return nil; }

    // 3. Run inference
    id<MLFeatureProvider> result = [_model predictionFromFeatures:provider
                                                            error:&err];

    CVPixelBufferRelease(pb);

    if (!result) { NSLog(@"Inference error: %@", err); return nil; }

    // 4. Collect all MLMultiArray outputs
    NSMutableDictionary* outputs = [NSMutableDictionary new];
    for (NSString* key in result.featureNames) {
        MLFeatureValue* val = [result featureValueForName:key];
        if (val.type == MLFeatureTypeMultiArray)
            outputs[key] = val.multiArrayValue;
    }
    return outputs;
}

@end

// ── C++ InferenceEngine — hides all ObjC from callers ──────

InferenceEngine::InferenceEngine(const std::string& path)
    : objc_engine_(nullptr), loaded_(false) {
    NSString* ns = [NSString stringWithUTF8String:path.c_str()];
    ObjCEngine* e = [[ObjCEngine alloc] initWithPath:ns];
    if (e.model) {
        objc_engine_ = (__bridge_retained void*)e;
        loaded_ = true;
    }
}

InferenceEngine::~InferenceEngine() {
    if (objc_engine_) { CFRelease(objc_engine_); objc_engine_ = nullptr; }
}

InferenceResult InferenceEngine::predict(const uint8_t* rgb, int w, int h) {
    InferenceResult result;
    result.success = false;

    auto t0 = std::chrono::high_resolution_clock::now();

    ObjCEngine* e = (__bridge ObjCEngine*)objc_engine_;
    NSDictionary* outputs = [e runWithRGB:rgb width:w height:h];

    auto t1 = std::chrono::high_resolution_clock::now();
    result.inference_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!outputs) return result;

    for (NSString* key in outputs) {
        MLMultiArray* arr = outputs[key];
        ActivationTensor t;
        t.name = [key UTF8String];
        for (NSNumber* d in arr.shape)
            t.shape.push_back(d.longLongValue);
        int64_t n = arr.count;
        t.data.resize(n);


        // NEW — checks actual dtype, handles float16 and float32
        MLMultiArrayDataType dtype = arr.dataType;

        // Log dtype on first tensor for debugging
        static bool dtype_logged = false;
        if (!dtype_logged) {
            NSLog(@"[Engine] MLMultiArray dtype: %ld (Float32=65568, Float16=65552)",
                (long)dtype);
            dtype_logged = true;
        }

        if (dtype == MLMultiArrayDataTypeFloat32) {
            std::copy((float*)arr.dataPointer,
                    (float*)arr.dataPointer + n,
                    t.data.begin());
        } else if (dtype == MLMultiArrayDataTypeFloat16) {
            // float16 → float32 conversion using Apple __fp16 (arm64 native)
            __fp16* src = (__fp16*)arr.dataPointer;
            for (int64_t i = 0; i < n; i++)
                t.data[i] = (float)src[i];
        } else if (dtype == MLMultiArrayDataTypeDouble) {
            double* src = (double*)arr.dataPointer;
            for (int64_t i = 0; i < n; i++)
                t.data[i] = (float)src[i];
        } else {
            NSLog(@"[Engine] Unknown dtype %ld — skipping tensor", (long)dtype);
            t.data.assign(n, 0.0f);
        }


        result.tensors.push_back(std::move(t));
    }

    result.success = true;
    return result;
}