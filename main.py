import pygame
import color
import key

from pygame import Surface
from modules.nn import *
from modules.mnist import MNIST

pygame.init()

dataset = MNIST()
network = NeuralNetwork('saves/digit_recognition.json')
screen = pygame.display.set_mode((500, 750))
clock = pygame.time.Clock()
font70 = pygame.font.SysFont('arial', 70)
font30 = pygame.font.SysFont('arial', 30)
 
def get_data_point_img(data_point: DataPoint) -> Surface:
    img = Surface((28, 28))

    for x in range(28):
        for y in range(28):
            pixel_brightness = int(data_point.inputs[y * 28 + x])
            img.set_at((x, y), (pixel_brightness, pixel_brightness, pixel_brightness))
    
    return img


def get_new_data_point() -> None:
    # get info
    new_data_point = dataset.get_random_image()
    data_point_img = pygame.transform.scale_by(get_data_point_img(new_data_point), 10)
    
    network_output = network.process(new_data_point.inputs)
    network_guess = network_output.index(max(network_output))
    actual_value = new_data_point.expected.index(max(new_data_point.expected))
    correct = network_guess == actual_value

    # render
    screen.fill(color.black)

    text = font70.render('Image', True, (255, 127, 0))
    rect = text.get_rect(center=(250, 50))
    screen.blit(text, rect)
    pygame.draw.rect(screen, color.grey, (100, 100, 300, 300))
    screen.blit(data_point_img, (110, 110))

    text = font30.render('Guess', True, (255, 127, 0))
    rect = text.get_rect(center=(150, 525))
    screen.blit(text, rect)
    text = font70.render(f'{network_guess}', True, color.green if correct else color.red)
    rect = text.get_rect(center=(150, 600))
    pygame.draw.rect(screen, color.green if correct else color.red, rect.inflate(40, 10), 10)
    screen.blit(text, rect)

    text = font30.render('Actual', True, (255, 127, 0))
    rect = text.get_rect(center=(350, 525))
    screen.blit(text, rect)
    text = font70.render(f'{actual_value}', True, color.blue)
    rect = text.get_rect(center=(350, 600))
    pygame.draw.rect(screen, color.blue, rect.inflate(40, 10), 10)
    screen.blit(text, rect)

    pygame.display.flip()


get_new_data_point()

running = True


while running:
    clock.tick(165)

    for event in pygame.event.get():
        match event.type:
            case pygame.QUIT:
                running = False
            case pygame.KEYDOWN:
                match event.key:
                    case key.space:
                        get_new_data_point()


pygame.quit()
