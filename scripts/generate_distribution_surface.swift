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

func lerp(_ a: CGFloat, _ b: CGFloat, _ t: CGFloat) -> CGFloat {
    a + (b - a) * t
}

func paletteColor(_ value: Double) -> (CGFloat, CGFloat, CGFloat) {
    let clamped = max(0.0, min(1.0, value))
    if clamped < 0.25 {
        let t = CGFloat(clamped / 0.25)
        return (lerp(0.06, 0.00, t), lerp(0.12, 0.55, t), lerp(0.36, 0.82, t))
    } else if clamped < 0.50 {
        let t = CGFloat((clamped - 0.25) / 0.25)
        return (lerp(0.00, 0.10, t), lerp(0.55, 0.80, t), lerp(0.82, 0.42, t))
    } else if clamped < 0.75 {
        let t = CGFloat((clamped - 0.50) / 0.25)
        return (lerp(0.10, 0.98, t), lerp(0.80, 0.82, t), lerp(0.42, 0.18, t))
    } else {
        let t = CGFloat((clamped - 0.75) / 0.25)
        return (lerp(0.98, 0.82, t), lerp(0.82, 0.12, t), lerp(0.18, 0.12, t))
    }
}

struct Point3D {
    let x: Double
    let c: Double
    let f: Double
}

guard CommandLine.arguments.count == 2 else {
    fputs("usage: swift generate_distribution_surface.swift <output_dir>\n", stderr)
    exit(1)
}

let outputDir = CommandLine.arguments[1]
let xPath = (outputDir as NSString).appendingPathComponent("x_cells.txt")
let cPath = (outputDir as NSString).appendingPathComponent("velocity_axis.txt")
let fPath = (outputDir as NSString).appendingPathComponent("distribution_f.txt")

let fm = FileManager.default
guard fm.fileExists(atPath: xPath),
      fm.fileExists(atPath: cPath),
      fm.fileExists(atPath: fPath) else {
    fputs("Missing required files for surface plot.\n", stderr)
    exit(1)
}

let x = try loadVector(from: xPath)
let c = try loadVector(from: cPath)
let values = try loadMatrix(from: fPath)

let width = 1460
let height = 980
let imageURL = URL(fileURLWithPath: (outputDir as NSString).appendingPathComponent("distribution_f_surface.png"))
let imageURLAlt = URL(fileURLWithPath: (outputDir as NSString).appendingPathComponent("distribution_f_surface_front.png"))

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

ctx.setFillColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
ctx.translateBy(x: 0, y: CGFloat(height))
ctx.scaleBy(x: 1, y: -1)
ctx.setAllowsAntialiasing(true)

let xMin = x.first ?? 0.0
let xMax = x.last ?? 1.0
let cMin = c.first ?? -1.0
let cMax = c.last ?? 1.0
let fMax = max(values.flatMap { $0 }.max() ?? 1.0, 1e-12)

func project(_ p: Point3D) -> CGPoint {
    let nx = (p.x - xMin) / max(xMax - xMin, 1e-12)
    let nc = (p.c - cMin) / max(cMax - cMin, 1e-12)
    let nf = p.f / fMax

    // Match the reference view:
    // - x runs along the front edge from left to right
    // - c runs along the left diagonal toward the viewer
    // - f is vertical
    // Using cMax as the front edge puts the visible front axis in x.
    let frontness = nc
    let px = 140.0 + nx * 1010.0 + frontness * 250.0
    let py = 760.0 - nx * 28.0 + frontness * 88.0 - nf * 440.0
    return CGPoint(x: px, y: py)
}

func strokeLine(_ a: CGPoint, _ b: CGPoint, color: (CGFloat, CGFloat, CGFloat), width: CGFloat, alpha: CGFloat = 1.0) {
    ctx.beginPath()
    ctx.move(to: a)
    ctx.addLine(to: b)
    ctx.setStrokeColor(red: color.0, green: color.1, blue: color.2, alpha: alpha)
    ctx.setLineWidth(width)
    ctx.strokePath()
}

func fillQuad(_ a: CGPoint, _ b: CGPoint, _ c: CGPoint, _ d: CGPoint, color: (CGFloat, CGFloat, CGFloat)) {
    ctx.beginPath()
    ctx.move(to: a)
    ctx.addLine(to: b)
    ctx.addLine(to: c)
    ctx.addLine(to: d)
    ctx.closePath()
    ctx.setFillColor(red: color.0, green: color.1, blue: color.2, alpha: 0.95)
    ctx.fillPath()
}

let floorA = project(Point3D(x: xMin, c: cMin, f: 0.0))
let floorB = project(Point3D(x: xMax, c: cMin, f: 0.0))
let floorC = project(Point3D(x: xMax, c: cMax, f: 0.0))
let floorD = project(Point3D(x: xMin, c: cMax, f: 0.0))
ctx.setFillColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
ctx.beginPath()
ctx.move(to: floorA)
ctx.addLine(to: floorB)
ctx.addLine(to: floorC)
ctx.addLine(to: floorD)
ctx.closePath()
ctx.fillPath()

for i in stride(from: values.count - 2, through: 0, by: -1) {
    for j in stride(from: values[i].count - 2, through: 0, by: -1) {
        let p00 = project(Point3D(x: x[i], c: c[j], f: values[i][j]))
        let p10 = project(Point3D(x: x[i + 1], c: c[j], f: values[i + 1][j]))
        let p11 = project(Point3D(x: x[i + 1], c: c[j + 1], f: values[i + 1][j + 1]))
        let p01 = project(Point3D(x: x[i], c: c[j + 1], f: values[i][j + 1]))
        let avg = (values[i][j] + values[i + 1][j] + values[i + 1][j + 1] + values[i][j + 1]) / (4.0 * fMax)
        fillQuad(p00, p10, p11, p01, color: paletteColor(avg))
    }
}

let meshColor: (CGFloat, CGFloat, CGFloat) = (0.18, 0.18, 0.20)
let xStride = max(1, x.count / 18)
let cStride = max(1, c.count / 16)
for i in stride(from: 0, to: x.count, by: xStride) {
    for j in 0..<(c.count - 1) {
        let a = project(Point3D(x: x[i], c: c[j], f: values[i][j]))
        let b = project(Point3D(x: x[i], c: c[j + 1], f: values[i][j + 1]))
        strokeLine(a, b, color: meshColor, width: 0.35, alpha: 0.05)
    }
}
for j in stride(from: 0, to: c.count, by: cStride) {
    for i in 0..<(x.count - 1) {
        let a = project(Point3D(x: x[i], c: c[j], f: values[i][j]))
        let b = project(Point3D(x: x[i + 1], c: c[j], f: values[i + 1][j]))
        strokeLine(a, b, color: meshColor, width: 0.35, alpha: 0.04)
    }
}

let axisColor: (CGFloat, CGFloat, CGFloat) = (0.08, 0.08, 0.10)
let zTop = project(Point3D(x: xMin, c: cMin, f: fMax))
strokeLine(floorD, floorC, color: axisColor, width: 2.4)
strokeLine(floorA, floorD, color: axisColor, width: 2.4)
strokeLine(floorA, zTop, color: axisColor, width: 2.4)

for i in 0...5 {
    let xv = xMin + Double(i) / 5.0 * (xMax - xMin)
    let p = project(Point3D(x: xv, c: cMax, f: 0.0))
    strokeLine(p, CGPoint(x: p.x, y: p.y + 10), color: axisColor, width: 1.0)
    drawText(ctx, String(format: "%.1f", xv), x: p.x, y: CGFloat(height) - p.y + 34, size: 15, align: .center)
}
for i in 0...5 {
    let cv = cMin + Double(i) / 5.0 * (cMax - cMin)
    let p = project(Point3D(x: xMin, c: cv, f: 0.0))
    strokeLine(p, CGPoint(x: p.x - 10, y: p.y + 4), color: axisColor, width: 1.0)
    drawText(ctx, String(format: "%.0f", cv), x: p.x - 16, y: CGFloat(height) - p.y + 8, size: 15, align: .right)
}
for i in 0...5 {
    let fv = Double(i) / 5.0 * fMax
    let p = project(Point3D(x: xMin, c: cMin, f: fv))
    strokeLine(p, CGPoint(x: p.x - 10, y: p.y), color: axisColor, width: 1.0)
    drawText(ctx, String(format: "%.1f", fv), x: p.x - 18, y: CGFloat(height) - p.y + 5, size: 15, align: .right)
}

drawText(ctx, "Distribution f(x, c)", x: CGFloat(width) / 2, y: 30, size: 30, align: .center)
drawText(ctx, "x", x: floorC.x + 26, y: CGFloat(height) - floorC.y + 12, size: 22, align: .left)
drawText(ctx, "c", x: floorA.x - 24, y: CGFloat(height) - floorA.y + 10, size: 22, align: .right)
drawText(ctx, "f", x: zTop.x - 18, y: CGFloat(height) - zTop.y - 10, size: 22, align: .right)

guard let image = ctx.makeImage(),
      let destination = CGImageDestinationCreateWithURL(imageURL as CFURL, UTType.png.identifier as CFString, 1, nil),
      let destinationAlt = CGImageDestinationCreateWithURL(imageURLAlt as CFURL, UTType.png.identifier as CFString, 1, nil) else {
    fputs("Could not finalize surface plot image.\n", stderr)
    exit(1)
}

CGImageDestinationAddImage(destination, image, nil)
CGImageDestinationAddImage(destinationAlt, image, nil)
guard CGImageDestinationFinalize(destination),
      CGImageDestinationFinalize(destinationAlt) else {
    fputs("Failed to write surface plot image.\n", stderr)
    exit(1)
}

print(imageURL.path)
print(imageURLAlt.path)
