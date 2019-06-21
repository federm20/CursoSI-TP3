from PIL import Image, ImageDraw
import random

# draw rectangles
for i in range(100):
    image = Image.new('RGB', (28, 28), 'white')  # could also open an existing image here to draw shapes over it
    draw = ImageDraw.Draw(image)
    x0 = random.randint(1,14)
    y0 = random.randint(1,14)
    x1 = random.randint(14,28)
    y1 = random.randint(14,28)
    draw.rectangle((x0, y0, x1, y1), fill='black', outline='red')  # can vary this bit to draw different shapes in different positions
    image.save('./train-shapes/rectangle/rectangle' + str(i) + '.png')

# draw ellipse
for i in range(100):
    image = Image.new('RGB', (28, 28), 'white')  # could also open an existing image here to draw shapes over it
    draw = ImageDraw.Draw(image)
    x0 = random.randint(1,14)
    y0 = random.randint(1,14)
    x1 = random.randint(14,28)
    y1 = random.randint(14,28)
    draw.ellipse((x0, y0, x1, y1), fill='black', outline='red')  # can vary this bit to draw different shapes in different positions
    image.save('./train-shapes/ellipse/ellipse' + str(i) + '.png')

# draw circle
for i in range(100):
    image = Image.new('RGB', (28, 28), 'white')  # could also open an existing image here to draw shapes over it
    draw = ImageDraw.Draw(image)
    x0 = random.randint(10,10)
    y0 = random.randint(10,10)
    dif = random.randint(5,12)
    x1 = x0 + dif
    y1 = y0 + dif
    draw.ellipse((x0, y0, x1, y1), fill='black', outline='red')  # can vary this bit to draw different shapes in different positions
    image.save('./train-shapes/circle/circle' + str(i) + '.png')

# draw square
for i in range(100):
    image = Image.new('RGB', (28, 28), 'white')  # could also open an existing image here to draw shapes over it
    draw = ImageDraw.Draw(image)
    x0 = random.randint(1,14)
    y0 = random.randint(1,14)
    dif = random.randint(1,14)
    x1 = x0 + dif
    y1 = y0 + dif
    draw.rectangle((x0, y0, x1, y1), fill='black', outline='red')  # can vary this bit to draw different shapes in different positions
    image.save('./train-shapes/square/square' + str(i) + '.png')

# draw triangle
for i in range(100):
    image = Image.new('RGB', (28, 28), 'white')  # could also open an existing image here to draw shapes over it
    draw = ImageDraw.Draw(image)
    x0 = random.randint(1,27)
    y0 = random.randint(1,27)
    x1 = random.randint(1,27)
    y1 = random.randint(1,27)
    x2 = random.randint(1,27)
    y2 = random.randint(1,27)
    draw.polygon([(x0, y0), (x1, y1), (x2, y2)], fill='black', outline='red')  # can vary this bit to draw different shapes in different positions
    image.save('./train-shapes/triangle/triangle' + str(i) + '.png')