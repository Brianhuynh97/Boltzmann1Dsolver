import Foundation
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers

func loadVector(from path: String) throws -> [Double] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    return text.split { $0.isWhitespace }.compactMap { Double($0) }
}

func drawText(
    _ ctx: CGContext,
    _ text: String,
    x: CGFloat,
    y: CGFloat,
    size: CGFloat,
    align: CTTextAlignment = .left
) {
    let font = CTFontCreateWithName("Helvetica" as CFString, size, nil)
    let color = CGColor(gray: 0.08, alpha: 1.0)
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

func strokeLine(_ ctx: CGContext, _ a: CGPoint, _ b: CGPoint, color: CGColor, width: CGFloat) {
    ctx.beginPath()
    ctx.move(to: a)
    ctx.addLine(to: b)
    ctx.setStrokeColor(color)
    ctx.setLineWidth(width)
    ctx.strokePath()
}

func renderProfile(
    x: [Double],
    y: [Double],
    title: String,
    xLabel: String,
    yLabel: String,
    outputURL: URL
) throws {
    let width = 980
    let height = 620
    let leftMargin: CGFloat = 90
    let rightMargin: CGFloat = 28
    let topMargin: CGFloat = 54
    let bottomMargin: CGFloat = 72
    let plotWidth = CGFloat(width) - leftMargin - rightMargin
    let plotHeight = CGFloat(height) - topMargin - bottomMargin

    guard let ctx = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        throw NSError(domain: "plot", code: 1)
    }

    ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
    ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
    ctx.translateBy(x: 0, y: CGFloat(height))
    ctx.scaleBy(x: 1, y: -1)
    ctx.setAllowsAntialiasing(true)

    let frame = CGRect(x: leftMargin, y: topMargin, width: plotWidth, height: plotHeight)
    ctx.setFillColor(red: 0.998, green: 0.998, blue: 0.998, alpha: 1)
    ctx.fill(frame)

    let xMin = x.first ?? 0
    let xMax = x.last ?? 1
    let yMin0 = y.min() ?? 0
    let yMax0 = y.max() ?? 1
    let span = max(yMax0 - yMin0, 1e-12)
    let yPad = 0.08 * span
    let yMin = yMin0 - yPad
    let yMax = yMax0 + yPad

    for i in 0...5 {
        let yf = CGFloat(i) / 5.0
        let gy = frame.maxY - yf * frame.height
        strokeLine(ctx, CGPoint(x: frame.minX, y: gy), CGPoint(x: frame.maxX, y: gy), color: CGColor(gray: 0.88, alpha: 1), width: 0.8)
        let yValue = yMin + Double(yf) * (yMax - yMin)
        drawText(ctx, String(format: "%.3g", yValue), x: frame.minX - 14, y: gy + 5, size: 14, align: .right)
    }
    for i in 0...5 {
        let xf = CGFloat(i) / 5.0
        let gx = frame.minX + xf * frame.width
        strokeLine(ctx, CGPoint(x: gx, y: frame.minY), CGPoint(x: gx, y: frame.maxY), color: CGColor(gray: 0.92, alpha: 1), width: 0.8)
        let xValue = xMin + Double(xf) * (xMax - xMin)
        drawText(ctx, String(format: "%.1f", xValue), x: gx, y: frame.maxY + 22, size: 14, align: .center)
    }

    let axisColor = CGColor(gray: 0.08, alpha: 1)
    strokeLine(ctx, CGPoint(x: frame.minX, y: frame.maxY), CGPoint(x: frame.minX, y: frame.minY), color: axisColor, width: 1.5)
    strokeLine(ctx, CGPoint(x: frame.minX, y: frame.maxY), CGPoint(x: frame.maxX, y: frame.maxY), color: axisColor, width: 1.5)

    ctx.beginPath()
    for i in 0..<min(x.count, y.count) {
        let xn = (x[i] - xMin) / max(xMax - xMin, 1e-12)
        let yn = (y[i] - yMin) / max(yMax - yMin, 1e-12)
        let point = CGPoint(
            x: frame.minX + CGFloat(xn) * frame.width,
            y: frame.maxY - CGFloat(yn) * frame.height
        )
        if i == 0 {
            ctx.move(to: point)
        } else {
            ctx.addLine(to: point)
        }
    }
    ctx.setStrokeColor(CGColor(red: 0.05, green: 0.05, blue: 0.05, alpha: 1))
    ctx.setLineWidth(2.4)
    ctx.strokePath()

    drawText(ctx, title, x: CGFloat(width) / 2, y: 28, size: 28, align: .center)
    drawText(ctx, xLabel, x: frame.midX, y: frame.maxY + 50, size: 18, align: .center)
    drawText(ctx, yLabel, x: 34, y: frame.midY + 6, size: 18)

    guard let image = ctx.makeImage(),
          let destination = CGImageDestinationCreateWithURL(outputURL as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw NSError(domain: "plot", code: 2)
    }
    CGImageDestinationAddImage(destination, image, nil)
    if !CGImageDestinationFinalize(destination) {
        throw NSError(domain: "plot", code: 3)
    }
}

func catmullRomSmoothedPoints(x: [Double], y: [Double], samplesPerSegment: Int) -> [(Double, Double)] {
    guard x.count == y.count, x.count >= 2 else { return zip(x, y).map { ($0.0, $0.1) } }

    func value(_ array: [Double], _ index: Int) -> Double {
        let clamped = max(0, min(index, array.count - 1))
        return array[clamped]
    }

    var points: [(Double, Double)] = []
    for i in 0..<(x.count - 1) {
        let x0 = value(x, i - 1)
        let x1 = value(x, i)
        let x2 = value(x, i + 1)
        let x3 = value(x, i + 2)
        let y0 = value(y, i - 1)
        let y1 = value(y, i)
        let y2 = value(y, i + 1)
        let y3 = value(y, i + 2)

        for s in 0..<(i == x.count - 2 ? samplesPerSegment + 1 : samplesPerSegment) {
            let t = Double(s) / Double(samplesPerSegment)
            let t2 = t * t
            let t3 = t2 * t

            let px = 0.5 * (
                (2.0 * x1) +
                (-x0 + x2) * t +
                (2.0 * x0 - 5.0 * x1 + 4.0 * x2 - x3) * t2 +
                (-x0 + 3.0 * x1 - 3.0 * x2 + x3) * t3
            )
            let py = 0.5 * (
                (2.0 * y1) +
                (-y0 + y2) * t +
                (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t2 +
                (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t3
            )
            points.append((px, py))
        }
    }
    return points
}

guard CommandLine.arguments.count == 2 else {
    fputs("usage: swift generate_scalar_profiles.swift <output_dir>\n", stderr)
    exit(1)
}

let outputDir = CommandLine.arguments[1]
let x = try loadVector(from: (outputDir as NSString).appendingPathComponent("x_cells.txt"))
let rho = try loadVector(from: (outputDir as NSString).appendingPathComponent("density.txt"))
let temp = try loadVector(from: (outputDir as NSString).appendingPathComponent("temperature.txt"))
let vel = try loadVector(from: (outputDir as NSString).appendingPathComponent("bulk_vx.txt"))

let rhoURL = URL(fileURLWithPath: (outputDir as NSString).appendingPathComponent("density_x.png"))
let tempURL = URL(fileURLWithPath: (outputDir as NSString).appendingPathComponent("temperature_x.png"))
let velURL = URL(fileURLWithPath: (outputDir as NSString).appendingPathComponent("velocity_x.png"))
let combinedURL = URL(fileURLWithPath: (outputDir as NSString).appendingPathComponent("macro_x.png"))

try renderProfile(x: x, y: rho, title: "Density rho Along x", xLabel: "spatial coordinate x", yLabel: "rho", outputURL: rhoURL)
try renderProfile(x: x, y: vel, title: "Velocity v Along x", xLabel: "spatial coordinate x", yLabel: "v", outputURL: velURL)
try renderProfile(x: x, y: temp, title: "Temperature T Along x", xLabel: "spatial coordinate x", yLabel: "T", outputURL: tempURL)

func yRange(_ y: [Double]) -> (Double, Double) {
    let yMin0 = y.min() ?? 0
    let yMax0 = y.max() ?? 1
    let span = max(yMax0 - yMin0, 1e-12)
    let yPad = 0.08 * span
    return (yMin0 - yPad, yMax0 + yPad)
}

func drawSeriesPanel(
    _ ctx: CGContext,
    frame: CGRect,
    x: [Double],
    y: [Double],
    panelTitle: String,
    yLabel: String,
    xLabel: String?,
    color: CGColor
) {
    let xMin = x.first ?? 0
    let xMax = x.last ?? 1
    let (yMin, yMax) = yRange(y)

    ctx.setFillColor(red: 0.998, green: 0.998, blue: 0.998, alpha: 1)
    ctx.fill(frame)
    ctx.setStrokeColor(CGColor(gray: 0.84, alpha: 1))
    ctx.setLineWidth(1)
    ctx.stroke(frame)

    for i in 0...5 {
        let yf = CGFloat(i) / 5.0
        let gy = frame.maxY - yf * frame.height
        strokeLine(ctx, CGPoint(x: frame.minX, y: gy), CGPoint(x: frame.maxX, y: gy), color: CGColor(gray: 0.88, alpha: 1), width: 0.8)
        let yValue = yMin + Double(yf) * (yMax - yMin)
        drawText(ctx, String(format: "%.3g", yValue), x: frame.minX - 14, y: gy + 5, size: 13, align: .right)
    }
    for i in 0...5 {
        let xf = CGFloat(i) / 5.0
        let gx = frame.minX + xf * frame.width
        strokeLine(ctx, CGPoint(x: gx, y: frame.minY), CGPoint(x: gx, y: frame.maxY), color: CGColor(gray: 0.92, alpha: 1), width: 0.8)
        if xLabel != nil {
            let xValue = xMin + Double(xf) * (xMax - xMin)
            drawText(ctx, String(format: "%.1f", xValue), x: gx, y: frame.maxY + 20, size: 13, align: .center)
        }
    }

    let axisColor = CGColor(gray: 0.08, alpha: 1)
    strokeLine(ctx, CGPoint(x: frame.minX, y: frame.maxY), CGPoint(x: frame.minX, y: frame.minY), color: axisColor, width: 1.4)
    strokeLine(ctx, CGPoint(x: frame.minX, y: frame.maxY), CGPoint(x: frame.maxX, y: frame.maxY), color: axisColor, width: 1.4)

    ctx.beginPath()
    for i in 0..<min(x.count, y.count) {
        let xn = (x[i] - xMin) / max(xMax - xMin, 1e-12)
        let yn = (y[i] - yMin) / max(yMax - yMin, 1e-12)
        let point = CGPoint(
            x: frame.minX + CGFloat(xn) * frame.width,
            y: frame.maxY - CGFloat(yn) * frame.height
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

    drawText(ctx, panelTitle, x: frame.minX, y: frame.minY - 18, size: 18)
    drawText(ctx, yLabel, x: frame.minX - 52, y: frame.midY + 6, size: 17)
    if let xLabel {
        drawText(ctx, xLabel, x: frame.midX, y: frame.maxY + 44, size: 17, align: .center)
    }
}

let combinedWidth = 1040
let combinedHeight = 720
let leftMargin: CGFloat = 96
let rightMargin: CGFloat = 26
let topMargin: CGFloat = 72
let bottomMargin: CGFloat = 76
let plotWidth = CGFloat(combinedWidth) - leftMargin - rightMargin
let plotHeight = CGFloat(combinedHeight) - topMargin - bottomMargin

guard let ctx = CGContext(
    data: nil,
    width: combinedWidth,
    height: combinedHeight,
    bitsPerComponent: 8,
    bytesPerRow: 0,
    space: CGColorSpaceCreateDeviceRGB(),
    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
) else {
    throw NSError(domain: "plot", code: 4)
}

ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
ctx.fill(CGRect(x: 0, y: 0, width: combinedWidth, height: combinedHeight))
ctx.translateBy(x: 0, y: CGFloat(combinedHeight))
ctx.scaleBy(x: 1, y: -1)
ctx.setAllowsAntialiasing(true)

drawText(ctx, "Macroscopic Profiles Along x", x: CGFloat(combinedWidth) / 2, y: 28, size: 28, align: .center)

let frame = CGRect(x: leftMargin, y: topMargin, width: plotWidth, height: plotHeight)
ctx.setFillColor(red: 0.998, green: 0.998, blue: 0.998, alpha: 1)
ctx.fill(frame)
ctx.setStrokeColor(CGColor(gray: 0.84, alpha: 1))
ctx.setLineWidth(1)
ctx.stroke(frame)

let xMin = x.first ?? 0
let xMax = x.last ?? 1

let normalizedSeries: [(name: String, values: [Double], color: CGColor)] = [
    ("rho", rho, CGColor(red: 0.08, green: 0.33, blue: 0.76, alpha: 1)),
    ("v", vel, CGColor(red: 0.12, green: 0.58, blue: 0.24, alpha: 1)),
    ("T", temp, CGColor(red: 0.86, green: 0.28, blue: 0.12, alpha: 1)),
].map { item in
    let minV = item.1.min() ?? 0
    let maxV = item.1.max() ?? 1
    let span = max(maxV - minV, 1e-12)
    let normalized = item.1.map { ($0 - minV) / span }
    return (item.0, normalized, item.2)
}

for i in 0...5 {
    let yf = CGFloat(i) / 5.0
    let gy = frame.maxY - yf * frame.height
    strokeLine(ctx, CGPoint(x: frame.minX, y: gy), CGPoint(x: frame.maxX, y: gy), color: CGColor(gray: 0.88, alpha: 1), width: 0.8)
    drawText(ctx, String(format: "%.1f", Double(yf)), x: frame.minX - 14, y: gy + 5, size: 13, align: .right)
}
for i in 0...5 {
    let xf = CGFloat(i) / 5.0
    let gx = frame.minX + xf * frame.width
    strokeLine(ctx, CGPoint(x: gx, y: frame.minY), CGPoint(x: gx, y: frame.maxY), color: CGColor(gray: 0.92, alpha: 1), width: 0.8)
    let xValue = xMin + Double(xf) * (xMax - xMin)
    drawText(ctx, String(format: "%.1f", xValue), x: gx, y: frame.maxY + 20, size: 13, align: .center)
}

let axisColor = CGColor(gray: 0.08, alpha: 1)
strokeLine(ctx, CGPoint(x: frame.minX, y: frame.maxY), CGPoint(x: frame.minX, y: frame.minY), color: axisColor, width: 1.4)
strokeLine(ctx, CGPoint(x: frame.minX, y: frame.maxY), CGPoint(x: frame.maxX, y: frame.maxY), color: axisColor, width: 1.4)

for series in normalizedSeries {
    let smoothed = catmullRomSmoothedPoints(x: x, y: series.values, samplesPerSegment: 8)
    ctx.beginPath()
    for i in 0..<smoothed.count {
        let xn = (smoothed[i].0 - xMin) / max(xMax - xMin, 1e-12)
        let yn = smoothed[i].1
        let point = CGPoint(
            x: frame.minX + CGFloat(xn) * frame.width,
            y: frame.maxY - CGFloat(yn) * frame.height
        )
        if i == 0 {
            ctx.move(to: point)
        } else {
            ctx.addLine(to: point)
        }
    }
    ctx.setStrokeColor(series.color)
    ctx.setLineWidth(3.0)
    ctx.strokePath()
}

drawText(ctx, "spatial coordinate x", x: frame.midX, y: frame.maxY + 46, size: 18, align: .center)

var legendX = frame.minX
let legendY = frame.minY - 28
for series in normalizedSeries {
    strokeLine(ctx, CGPoint(x: legendX, y: legendY), CGPoint(x: legendX + 28, y: legendY), color: series.color, width: 3)
    drawText(ctx, series.name, x: legendX + 38, y: legendY + 5, size: 16)
    legendX += 112
}

guard let combinedImage = ctx.makeImage(),
      let combinedDestination = CGImageDestinationCreateWithURL(combinedURL as CFURL, UTType.png.identifier as CFString, 1, nil) else {
    throw NSError(domain: "plot", code: 5)
}
CGImageDestinationAddImage(combinedDestination, combinedImage, nil)
if !CGImageDestinationFinalize(combinedDestination) {
    throw NSError(domain: "plot", code: 6)
}

print(rhoURL.path)
print(velURL.path)
print(tempURL.path)
print(combinedURL.path)
