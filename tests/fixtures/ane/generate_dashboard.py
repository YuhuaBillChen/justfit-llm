"""Generate the fictional 1920x1088 OCR fixture; no private data or performance claims."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--font', type=Path, required=True, help='DejaVuSans.ttf')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    im = Image.new('RGB', (1920, 1088), '#101827')
    d = ImageDraw.Draw(im)
    def text(x, y, value, size=26, color='#e8edf5'):
        d.text((x, y), value, font=ImageFont.truetype(str(a.font), size), fill=color)
    text(100, 55, 'COMMUNITY GARDEN / SAMPLE OPERATIONS', 36)
    text(100, 118, 'Fictional demonstration data. All names and counts are synthetic.', 25, '#9fafc7')
    cards = [('OPEN ORDERS', '73'), ('COMPLETED ORDERS', '184'), ('ACTIVE VOLUNTEERS', '26'), ('WATER TANK', '620 L')]
    for i, (label, value) in enumerate(cards):
        x = 100 + i * 435
        d.rounded_rectangle((x, 195, x+405, 365), radius=16, fill='#22324b')
        text(x+24, 218, label, 23, '#a9bedb')
        text(x+24, 268, value, 48, '#69d9c1')
    text(100, 405, 'Harvest packing schedule', 30)
    columns = [125, 490, 910, 1310]
    for x, label in zip(columns, ['AREA', 'CROP', 'CRATES READY', 'PICKUP']):
        text(x, 467, label, 23, '#8ca9cf')
    rows = [('NORTH', 'Tomatoes', '42', '09:30'), ('EAST', 'Carrots', '67', '10:15'), ('SOUTH', 'Peppers', '51', '11:00')]
    for i, row in enumerate(rows):
        y = 520+i*86
        d.rectangle((100, y-8, 1810, y+62), fill='#1b2940')
        for x, value in zip(columns, row):
            text(x, y+7, value, 29)
    text(100, 818, 'Dispatch notes', 30)
    text(125, 880, 'Use reusable crates. Confirm the pickup time before loading.', 26)
    text(125, 931, 'Keep walkways clear and return unused labels to the packing desk.', 26)
    text(100, 1022, 'DEMO FIXTURE  |  No real accounts, hosts, customers or benchmark results', 23, '#9fafc7')
    im.save(a.output, format='PNG')


if __name__ == '__main__':
    main()
