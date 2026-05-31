import CoreML
import Foundation

guard CommandLine.arguments.count > 1 else {
    print("{\"error\": \"Usage: compute_plan <path.mlpackage>\"}")
    exit(1)
}

let pkgURL   = URL(fileURLWithPath: CommandLine.arguments[1])
let sema     = DispatchSemaphore(value: 0)
var jsonOut  = "{\"error\": \"unknown\"}"

Task {
    defer { sema.signal() }

    // Step 1: compile .mlpackage → .mlmodelc
    let compiledURL: URL
    do {
        compiledURL = try MLModel.compileModel(at: pkgURL)
    } catch {
        jsonOut = "{\"error\": \"compile failed: \(error.localizedDescription)\"}"
        return
    }

    guard #available(macOS 14.0, *) else {
        jsonOut = "{\"error\": \"macOS 14+ required\"}"
        return
    }

    // Step 2: load compute plan (async throws — NOT a callback)
    let cfg = MLModelConfiguration()
    cfg.computeUnits = .all

    do {
        let plan = try await MLComputePlan.load(
            contentsOf: compiledURL,
            configuration: cfg
        )

        var ops: [[String: String]] = []

        if case .program(let prog) = plan.modelStructure {
            for (_, fn) in prog.functions {
                for block in fn.blocks {
                    for op in block.operations {
                        let usage  = plan.computeDeviceUsage(for: op)
                        var device = "unknown"
                        if let pref = usage?.preferredComputeDevice {
                            if      pref is MLNeuralEngineComputeDevice { device = "ANE" }
                            else if pref is MLGPUComputeDevice           { device = "GPU" }
                            else if pref is MLCPUComputeDevice           { device = "CPU" }
                        }
                        ops.append(["op": op.operatorName, "device": device])
                    }
                }
            }
        }

        let result: [String: Any] = ["ops": ops, "total": ops.count]
        if let data = try? JSONSerialization.data(
                withJSONObject: result, options: .prettyPrinted),
           let str  = String(data: data, encoding: .utf8) {
            jsonOut = str
        }

    } catch {
        jsonOut = "{\"error\": \"plan load failed: \(error.localizedDescription)\"}"
    }
}

sema.wait()
print(jsonOut)