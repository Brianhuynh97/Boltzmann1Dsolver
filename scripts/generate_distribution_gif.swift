import Foundation
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers

func loadVector(from path: String) throws -> [Double] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    return text
        .split { $0.isWhitespace }
        .compactMap { Double($0) }
}

func loadMatrix(from path: String) throws -> [[Double]] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    return text
        .split(whereSeparator: \.isNewline)
        .map { line in
            line.split { $0.isWhitespace }.compactMap { Double($0) }
        }
}

func loadTimes(from path: String) throws -> [Double] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    return text
        .split(whereSeparator: \.isNewline)
        .compactMap { line -> Double? in
            let parts = line.split { $0.isWhitespace }
            guard parts.count >= 2 else { return nil }
            return Double(parts[1])
        }
}

func sampleIndices(size: Int, count: Int) -> [Int] {
    guard size > 0 else { return [] }
    if count >= size { return Array(0..<size) }
    var result: [Int] = []
    for i in 0..<count {
        let idx = Int(round(Double(i) * Double(size - 1) / Double(max(count - 1, 1))))
        if result.last != idx {
            result.append(idx)
        }
    }
    return result
}

func setStroke(_ ctx: CGContext, _ color: (CGFloat, CGFloat, CGFloat), _ width: CGFloat) {
    ctx.setStrokeColor(red: color.0, green: color.1, blue: color.2, alpha: 1.0)
    ctx.setLineWidth(width)
}

func drawLine(_ ctx: CGContext, _ x1: CGFloat, _ y1: CGFloat, _ x2: CGFloat, _ y2: CGFloat) {
    ctx.beginPath()
    ctx.move(to: CGPoint(x: x1, y: y1))
    ctx.addLine(to: CGPoint(x: x2, y: y2))
    ctx.strokePath()
}

enum TextAlign {
    case left
    case center
    case right
}

func drawText(
    _ ctx: CGContext,
    _ text: String,
    x: CGFloat,
    y: CGFloat,
    size: CGFloat,
    canvasHeight: Int,
    align: TextAlign = .left
) {
    let font = CTFontCreateWithName("Helvetica" as CFString, size, nil)
    let color = CGColor(gray: 0.12, alpha: 1.0)
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
    switch align {
    case .center:
        drawX -= bounds.width / 2
    case .right:
        drawX -= bounds.width
    case .left:
        break
    }

    ctx.saveGState()
    ctx.textMatrix = .identity
    ctx.translateBy(x: drawX, y: y)
    ctx.scaleBy(x: 1, y: -1)
    ctx.textPosition = .zero
    CTLineDraw(line, ctx)
    ctx.restoreGState()
}

guard CommandLine.arguments.count == 2 else {
    fputs("usage: swift generate_distribution_gif.swift <output_dir>\n", stderr)
    exit(1)
}

let outputDir = CommandLine.arguments[1]
let snapshotDir = (outputDir as NSString).appendingPathComponent("snapshots")
let xPath = (outputDir as NSString).appendingPathComponent("x_cells.txt")
let vPath = (outputDir as NSString).appendingPathComponent("velocity_axis.txt")
let timesPath = (snapshotDir as NSString).appendingPathComponent("snapshot_times.txt")

let fileManager = FileManager.default
guard fileManager.fileExists(atPath: snapshotDir),
      fileManager.fileExists(atPath: xPath),
      fileManager.fileExists(atPath: vPath),
      fileManager.fileExists(atPath: timesPath) else {
    fputs("Missing required files for GIF animation.\n", stderr)
    exit(1)
}

let frameURLs = try fileManager.contentsOfDirectory(at: URL(fileURLWithPath: snapshotDir), includingPropertiesForKeys: nil)
    .filter { $0.lastPathComponent.hasPrefix("distribution_f_") && $0.pathExtension == "txt" }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }

guard !frameURLs.isEmpty else {
    fputs("No snapshot frames found.\n", stderr)
    exit(1)
}

let x = try loadVector(from: xPath)
let v = try loadVector(from: vPath)
let times = try loadTimes(from: timesPath)
let frames = try frameURLs.map { try loadMatrix(from: $0.path) }

let width = 800
let height = 500
let marginLeft: CGFloat = 70
let marginRight: CGFloat = 20
let marginTop: CGFloat = 20
let marginBottom: CGFloat = 40
let plotWidth = CGFloat(width) - marginLeft - marginRight
let plotHeight = CGFloat(height) - marginTop - marginBottom

let ids = sampleIndices(size: x.count, count: min(6, x.count))
let colors: [(CGFloat, CGFloat, CGFloat)] = [
    (13/255, 8/255, 135/255),
    (84/255, 3/255, 160/255),
    (139/255, 10/255, 165/255),
    (193/255, 45/255, 99/255),
    (240/255, 96/255, 39/255),
    (249/255, 201/255, 50/255),
]

let outURL = URL(fileURLWithPath: (outputDir as NSString).appendingPathComponent("distribution_f.gif"))
guard let destination = CGImageDestinationCreateWithURL(outURL as CFURL, UTType.gif.identifier as CFString, frames.count, nil) else {
    fputs("Could not create GIF destination.\n", stderr)
    exit(1)
}

let gifProperties: CFDictionary = [
    kCGImagePropertyGIFDictionary: [
        kCGImagePropertyGIFLoopCount: 0
    ]
] as CFDictionary
CGImageDestinationSetProperties(destination, gifProperties)

let colorSpace = CGColorSpaceCreateDeviceRGB()
let vMin = v.first ?? -1.0
let vMax = v.last ?? 1.0

func sx(_ value: Double) -> CGFloat {
    let denom = max(vMax - vMin, 1e-12)
    return marginLeft + CGFloat((value - vMin) / denom) * plotWidth
}

func sy(_ value: Double, _ fMax: Double) -> CGFloat {
    let denom = max(fMax, 1e-12)
    return CGFloat(height) - marginBottom - CGFloat(value / denom) * plotHeight
}

for (frameIndex, frame) in frames.enumerated() {
    guard let ctx = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        fputs("Could not create bitmap context.\n", stderr)
        exit(1)
    }

    ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
    ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
    ctx.setAllowsAntialiasing(true)
    ctx.translateBy(x: 0, y: CGFloat(height))
    ctx.scaleBy(x: 1, y: -1)

    setStroke(ctx, (0.85, 0.82, 0.78), 1)
    for i in 0...5 {
        let fx = marginLeft + CGFloat(i) / 5.0 * plotWidth
        let fy = marginTop + CGFloat(i) / 5.0 * plotHeight
        drawLine(ctx, fx, marginTop, fx, CGFloat(height) - marginBottom)
        drawLine(ctx, marginLeft, fy, CGFloat(width) - marginRight, fy)
    }

    setStroke(ctx, (0.08, 0.08, 0.08), 1.6)
    drawLine(ctx, marginLeft, marginTop, marginLeft, CGFloat(height) - marginBottom)
    drawLine(ctx, marginLeft, CGFloat(height) - marginBottom, CGFloat(width) - marginRight, CGFloat(height) - marginBottom)

    let framePeak = frame.flatMap { $0 }.max() ?? 1.0
    let fMax = max(framePeak * 1.05, 1e-12)

    for i in 0...5 {
        let fx = marginLeft + CGFloat(i) / 5.0 * plotWidth
        let fy = CGFloat(height) - marginBottom - CGFloat(i) / 5.0 * plotHeight
        let vTick = vMin + Double(i) / 5.0 * (vMax - vMin)
        let fTick = Double(i) / 5.0 * fMax
        drawText(ctx, String(format: "%.1f", vTick), x: fx, y: CGFloat(height) - 12, size: 12, canvasHeight: height, align: .center)
        drawText(ctx, String(format: "%.2f", fTick), x: marginLeft - 10, y: fy + 4, size: 12, canvasHeight: height, align: .right)
    }

    let timeValue = frameIndex < times.count ? times[frameIndex] : 0.0
    drawText(ctx, String(format: "Distribution f(c) at Selected x, t=%.4f", timeValue), x: CGFloat(width) / 2, y: 18, size: 18, canvasHeight: height, align: .center)
    drawText(ctx, "velocity c", x: CGFloat(width) / 2, y: CGFloat(height) - 12, size: 15, canvasHeight: height, align: .center)
    drawText(ctx, "distribution f", x: 16, y: marginTop + 18, size: 15, canvasHeight: height, align: .left)

    for (curveIndex, sampleID) in ids.enumerated() {
        guard sampleID < frame.count else { continue }
        let profile = frame[sampleID]
        setStroke(ctx, colors[curveIndex % colors.count], 2.5)
        ctx.beginPath()
        for (j, cValue) in v.enumerated() {
            guard j < profile.count else { continue }
            let px = sx(cValue)
            let py = sy(profile[j], fMax)
            if j == 0 {
                ctx.move(to: CGPoint(x: px, y: py))
            } else {
                ctx.addLine(to: CGPoint(x: px, y: py))
            }
        }
        ctx.strokePath()
    }

    guard let image = ctx.makeImage() else {
        fputs("Could not create image frame.\n", stderr)
        exit(1)
    }

    let frameProperties: CFDictionary = [
        kCGImagePropertyGIFDictionary: [
            kCGImagePropertyGIFDelayTime: 0.26
        ]
    ] as CFDictionary
    CGImageDestinationAddImage(destination, image, frameProperties)
}

guard CGImageDestinationFinalize(destination) else {
    fputs("Failed to finalize GIF.\n", stderr)
    exit(1)
}

print(outURL.path)
