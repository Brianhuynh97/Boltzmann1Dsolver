import Foundation
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers

func loadVector(from path: String) throws -> [Double] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    return text.split { $0.isWhitespace }.compactMap { Double($0) }
}

func loadMatrix(from path: String) throws -> [[Double]] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    return text
        .split(whereSeparator: \.isNewline)
        .map { line in line.split { $0.isWhitespace }.compactMap { Double($0) } }
}

func loadTimes(from path: String) throws -> [(Int, Double)] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    return text
        .split(whereSeparator: \.isNewline)
        .compactMap { line in
            let parts = line.split { $0.isWhitespace }
            guard parts.count >= 2,
                  let frame = Int(parts[0]),
                  let time = Double(parts[1]) else {
                return nil
            }
            return (frame, time)
        }
}

func drawText(
    _ ctx: CGContext,
    _ text: String,
    x: CGFloat,
    y: CGFloat,
    size: CGFloat,
    color: CGColor = CGColor(gray: 0.08, alpha: 1.0),
    align: CTTextAlignment = .left
) {
    let font = CTFontCreateWithName("Helvetica" as CFString, size, nil)
    let attr = NSAttributedString(
        string: text,
        attributes: [
            NSAttributedString.Key(rawValue: kCTFontAttributeName as String): font,
            NSAttributedString.Key(rawValue: kCTForegroundColorAttributeName as String): color,
        ]
    )
    let line = CTLineCreateWithAttributedString(attr)
    let bounds = CTLineGetBoundsWithOptions(line, .useOpticalBounds)
    var drawX = x
    if align == .center { drawX -= bounds.width / 2 }
    if align == .right { drawX -= bounds.width }

    ctx.saveGState()
    ctx.textMatrix = .identity
    ctx.translateBy(x: drawX, y: y)
    ctx.scaleBy(x: 1, y: -1)
    CTLineDraw(line, ctx)
    ctx.restoreGState()
}

struct MacroProfile {
    let time: Double
    let density: [Double]
    let velocity: [Double]
    let temperature: [Double]
}

func computeMacroProfile(
    distribution: [[Double]],
    velocityAxis: [Double],
    particleMass: Double,
    boltzmannOverMass: Double
) -> (density: [Double], velocity: [Double], temperature: [Double]) {
    let dv = velocityAxis.count > 1 ? (velocityAxis[1] - velocityAxis[0]) : 1.0
    var density = Array(repeating: 0.0, count: distribution.count)
    var velocity = Array(repeating: 0.0, count: distribution.count)
    var temperature = Array(repeating: 0.0, count: distribution.count)

    for i in 0..<distribution.count {
        var rho = 0.0
        var rhoU = 0.0
        var energy = 0.0
        for j in 0..<distribution[i].count {
            let f = distribution[i][j]
            let c = velocityAxis[j]
            rho += particleMass * f * dv
            rhoU += particleMass * c * f * dv
            energy += 0.5 * particleMass * c * c * f * dv
        }
        let safeRho = max(rho, 1e-12)
        let u = rhoU / safeRho
        let temp = max((2.0 * energy / safeRho - u * u) / max(boltzmannOverMass, 1e-12), 1e-8)
        density[i] = rho
        velocity[i] = u
        temperature[i] = temp
    }
    return (density, velocity, temperature)
}

func nearestProfiles(
    snapshotTimes: [(Int, Double)],
    outputDir: String,
    velocityAxis: [Double],
    particleMass: Double,
    boltzmannOverMass: Double
) throws -> [MacroProfile] {
    let maxTime = snapshotTimes.map(\.1).max() ?? 0.0
    let requested = stride(from: 0.0, through: floor(maxTime + 1e-9), by: 1.0).map { $0 }
    var chosen = Set<Int>()
    var profiles: [MacroProfile] = []

    for target in requested {
        guard let nearest = snapshotTimes.min(by: { abs($0.1 - target) < abs($1.1 - target) }) else {
            continue
        }
        if chosen.contains(nearest.0) { continue }
        chosen.insert(nearest.0)
        let fileName = String(format: "distribution_f_%04d.txt", nearest.0)
        let path = (outputDir as NSString).appendingPathComponent("snapshots/\(fileName)")
        let matrix = try loadMatrix(from: path)
        let macro = computeMacroProfile(
            distribution: matrix,
            velocityAxis: velocityAxis,
            particleMass: particleMass,
            boltzmannOverMass: boltzmannOverMass
        )
        profiles.append(
            MacroProfile(
                time: nearest.1,
                density: macro.density,
                velocity: macro.velocity,
                temperature: macro.temperature
            )
        )
    }

    return profiles.sorted { $0.time < $1.time }
}

func strokeLine(_ ctx: CGContext, _ a: CGPoint, _ b: CGPoint, color: CGColor, width: CGFloat) {
    ctx.beginPath()
    ctx.move(to: a)
    ctx.addLine(to: b)
    ctx.setStrokeColor(color)
    ctx.setLineWidth(width)
    ctx.strokePath()
}

guard CommandLine.arguments.count == 2 else {
    fputs("usage: swift generate_macro_profiles.swift <output_dir>\n", stderr)
    exit(1)
}

let outputDir = CommandLine.arguments[1]
let xPath = (outputDir as NSString).appendingPathComponent("x_cells.txt")
let vPath = (outputDir as NSString).appendingPathComponent("velocity_axis.txt")
let tPath = (outputDir as NSString).appendingPathComponent("snapshots/snapshot_times.txt")

let x = try loadVector(from: xPath)
let velocityAxis = try loadVector(from: vPath)
let snapshotTimes = try loadTimes(from: tPath)

// Current solver defaults written from ChannelProblemData.
let particleMass = 1.0
let boltzmannOverMass = 0.5

let profiles = try nearestProfiles(
    snapshotTimes: snapshotTimes,
    outputDir: outputDir,
    velocityAxis: velocityAxis,
    particleMass: particleMass,
    boltzmannOverMass: boltzmannOverMass
)

guard !profiles.isEmpty else {
    fputs("No snapshot profiles available.\n", stderr)
    exit(1)
}

let width = 1280
let height = 980
let leftMargin: CGFloat = 110
let rightMargin: CGFloat = 36
let topMargin: CGFloat = 60
let bottomMargin: CGFloat = 70
let panelGap: CGFloat = 44
let legendHeight: CGFloat = 42
let panelWidth = CGFloat(width) - leftMargin - rightMargin
let panelHeight = (CGFloat(height) - topMargin - bottomMargin - legendHeight - 2 * panelGap) / 3.0

let outputURL = URL(fileURLWithPath: (outputDir as NSString).appendingPathComponent("macro_profiles_vs_x.png"))
guard let ctx = CGContext(
    data: nil,
    width: width,
    height: height,
    bitsPerComponent: 8,
    bytesPerRow: 0,
    space: CGColorSpaceCreateDeviceRGB(),
    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
) else {
    fputs("Could not create image context.\n", stderr)
    exit(1)
}

ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
ctx.translateBy(x: 0, y: CGFloat(height))
ctx.scaleBy(x: 1, y: -1)
ctx.setAllowsAntialiasing(true)

let palette: [CGColor] = [
    CGColor(red: 0.06, green: 0.24, blue: 0.64, alpha: 1),
    CGColor(red: 0.11, green: 0.62, blue: 0.74, alpha: 1),
    CGColor(red: 0.22, green: 0.71, blue: 0.29, alpha: 1),
    CGColor(red: 0.96, green: 0.62, blue: 0.05, alpha: 1),
    CGColor(red: 0.87, green: 0.27, blue: 0.12, alpha: 1),
    CGColor(red: 0.55, green: 0.18, blue: 0.68, alpha: 1),
    CGColor(red: 0.78, green: 0.11, blue: 0.43, alpha: 1),
    CGColor(red: 0.15, green: 0.15, blue: 0.15, alpha: 1),
]

func yRange(_ values: [[Double]]) -> (Double, Double) {
    let flat = values.flatMap { $0 }
    let minV = flat.min() ?? 0.0
    let maxV = flat.max() ?? 1.0
    if abs(maxV - minV) < 1e-12 {
        return (minV - 0.5, maxV + 0.5)
    }
    let pad = 0.08 * (maxV - minV)
    return (minV - pad, maxV + pad)
}

let densityRange = yRange(profiles.map(\.density))
let velocityRange = yRange(profiles.map(\.velocity))
let temperatureRange = yRange(profiles.map(\.temperature))
let xMin = x.first ?? 0.0
let xMax = x.last ?? Double(max(x.count - 1, 1))

func plotPanel(
    title: String,
    yLabel: String,
    profiles: [MacroProfile],
    keyPath: KeyPath<MacroProfile, [Double]>,
    range: (Double, Double),
    topY: CGFloat,
    showXAxis: Bool
) {
    let frame = CGRect(x: leftMargin, y: topY, width: panelWidth, height: panelHeight)

    ctx.setFillColor(red: 0.995, green: 0.995, blue: 0.998, alpha: 1)
    ctx.fill(frame)
    ctx.setStrokeColor(CGColor(gray: 0.82, alpha: 1))
    ctx.setLineWidth(1)
    ctx.stroke(frame)

    for i in 1...4 {
        let y = frame.minY + CGFloat(i) * frame.height / 5.0
        strokeLine(ctx, CGPoint(x: frame.minX, y: y), CGPoint(x: frame.maxX, y: y), color: CGColor(gray: 0.88, alpha: 1), width: 0.8)
    }
    for i in 1...5 {
        let xTick = frame.minX + CGFloat(i) * frame.width / 6.0
        strokeLine(ctx, CGPoint(x: xTick, y: frame.minY), CGPoint(x: xTick, y: frame.maxY), color: CGColor(gray: 0.92, alpha: 1), width: 0.7)
    }

    strokeLine(ctx, CGPoint(x: frame.minX, y: frame.maxY), CGPoint(x: frame.minX, y: frame.minY), color: CGColor(gray: 0.1, alpha: 1), width: 1.4)
    strokeLine(ctx, CGPoint(x: frame.minX, y: frame.minY), CGPoint(x: frame.maxX, y: frame.minY), color: CGColor(gray: 0.1, alpha: 1), width: 1.4)

    drawText(ctx, title, x: frame.minX, y: frame.minY - 28, size: 20)
    drawText(ctx, yLabel, x: 42, y: frame.midY + 6, size: 19)

    for i in 0...4 {
        let fraction = Double(i) / 4.0
        let value = range.0 + fraction * (range.1 - range.0)
        let y = frame.maxY - CGFloat(fraction) * frame.height
        drawText(ctx, String(format: "%.3g", value), x: frame.minX - 16, y: y + 5, size: 14, align: .right)
    }

    if showXAxis {
        drawText(ctx, "x", x: frame.midX, y: frame.maxY + 34, size: 18, align: .center)
        for i in 0...5 {
            let fraction = Double(i) / 5.0
            let value = xMin + fraction * (xMax - xMin)
            let xPixel = frame.minX + CGFloat(fraction) * frame.width
            drawText(ctx, String(format: "%.1f", value), x: xPixel, y: frame.maxY + 18, size: 14, align: .center)
        }
    }

    for (index, profile) in profiles.enumerated() {
        let series = profile[keyPath: keyPath]
        let color = palette[index % palette.count]
        ctx.beginPath()
        for i in 0..<min(series.count, x.count) {
            let xNorm = (x[i] - xMin) / max(xMax - xMin, 1e-12)
            let yNorm = (series[i] - range.0) / max(range.1 - range.0, 1e-12)
            let point = CGPoint(
                x: frame.minX + CGFloat(xNorm) * frame.width,
                y: frame.maxY - CGFloat(yNorm) * frame.height
            )
            if i == 0 {
                ctx.move(to: point)
            } else {
                ctx.addLine(to: point)
            }
        }
        ctx.setStrokeColor(color)
        ctx.setLineWidth(2.4)
        ctx.strokePath()
    }
}

drawText(ctx, "Macroscopic Profiles vs x at Fixed Times", x: CGFloat(width) / 2, y: 24, size: 28, align: .center)

let densityTop = topMargin
let velocityTop = topMargin + panelHeight + panelGap
let temperatureTop = topMargin + 2 * (panelHeight + panelGap)

plotPanel(title: "Density", yLabel: "rho", profiles: profiles, keyPath: \.density, range: densityRange, topY: densityTop, showXAxis: false)
plotPanel(title: "Velocity", yLabel: "v", profiles: profiles, keyPath: \.velocity, range: velocityRange, topY: velocityTop, showXAxis: false)
plotPanel(title: "Temperature", yLabel: "T", profiles: profiles, keyPath: \.temperature, range: temperatureRange, topY: temperatureTop, showXAxis: true)

let legendY = CGFloat(height) - bottomMargin + 18
var legendX = leftMargin
for (index, profile) in profiles.enumerated() {
    let color = palette[index % palette.count]
    strokeLine(ctx, CGPoint(x: legendX, y: legendY), CGPoint(x: legendX + 26, y: legendY), color: color, width: 3)
    drawText(ctx, String(format: "t = %.2f", profile.time), x: legendX + 34, y: legendY + 5, size: 15)
    legendX += 118
}

guard let image = ctx.makeImage(),
      let destination = CGImageDestinationCreateWithURL(outputURL as CFURL, UTType.png.identifier as CFString, 1, nil) else {
    fputs("Could not finalize macro profile image.\n", stderr)
    exit(1)
}

CGImageDestinationAddImage(destination, image, nil)
guard CGImageDestinationFinalize(destination) else {
    fputs("Failed to write macro profile image.\n", stderr)
    exit(1)
}

print(outputURL.path)
