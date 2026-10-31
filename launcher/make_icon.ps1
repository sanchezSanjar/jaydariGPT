# Draws the JaydariGPT "J" logo and writes assets\jaydarigpt.ico (16-256 px, PNG entries).
Add-Type -AssemblyName System.Drawing

$root = Split-Path -Parent $PSScriptRoot
$assets = Join-Path $root "assets"
New-Item -ItemType Directory -Force $assets | Out-Null
$icoPath = Join-Path $assets "jaydarigpt.ico"

$sizes = 16, 24, 32, 48, 64, 128, 256
$images = @()
foreach ($size in $sizes) {
    $bmp = New-Object System.Drawing.Bitmap $size, $size
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $g.Clear([System.Drawing.Color]::Transparent)

    $accent = New-Object System.Drawing.SolidBrush ([System.Drawing.ColorTranslator]::FromHtml("#7c9cff"))
    $ink = New-Object System.Drawing.SolidBrush ([System.Drawing.ColorTranslator]::FromHtml("#0e1220"))
    $pad = [Math]::Max(0.5, $size * 0.03)
    $g.FillEllipse($accent, $pad, $pad, $size - 2 * $pad, $size - 2 * $pad)

    $font = New-Object System.Drawing.Font "Segoe UI", ([single]($size * 0.52)), ([System.Drawing.FontStyle]::Bold), ([System.Drawing.GraphicsUnit]::Pixel)
    $format = New-Object System.Drawing.StringFormat
    $format.Alignment = [System.Drawing.StringAlignment]::Center
    $format.LineAlignment = [System.Drawing.StringAlignment]::Center
    $box = New-Object System.Drawing.RectangleF 0, ([single]($size * 0.02)), $size, $size
    $g.DrawString("J", $font, $ink, $box, $format)
    $g.Dispose()

    $stream = New-Object System.IO.MemoryStream
    $bmp.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
    $images += , @($size, $stream.ToArray())
    $bmp.Dispose()
}

# ICO container: header, one directory entry per image, then the PNG data.
$out = New-Object System.IO.MemoryStream
$w = New-Object System.IO.BinaryWriter $out
$w.Write([UInt16]0); $w.Write([UInt16]1); $w.Write([UInt16]$images.Count)
$offset = 6 + 16 * $images.Count
foreach ($img in $images) {
    $dim = if ($img[0] -ge 256) { 0 } else { $img[0] }
    $w.Write([byte]$dim); $w.Write([byte]$dim); $w.Write([byte]0); $w.Write([byte]0)
    $w.Write([UInt16]1); $w.Write([UInt16]32)
    $w.Write([UInt32]$img[1].Length); $w.Write([UInt32]$offset)
    $offset += $img[1].Length
}
foreach ($img in $images) { $w.Write($img[1]) }
$w.Flush()
[System.IO.File]::WriteAllBytes($icoPath, $out.ToArray())
Write-Output "Icon written: $icoPath"
