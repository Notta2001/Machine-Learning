import numpy as np 
import pygame


def sigmoid(z) :
	return 1.0/(1 + np.exp(-z))

def split(p) : 
	if p >= 0.5 :
		return 1
	else : 
		return 0

def predict(features, weights) :
	z = np.dot(features, weights)
	return sigmoid(z)

def cost_function(features, labels, weights) :
	n = len(labels)
	predictions = predict(features, weights)
	cost_class1 = -labels*np.log(predictions)
	cost_class2 = -(1-labels)*np.log(1 - predictions)
	cost = cost_class1 + cost_class2
	return cost.sum()/n

def update_weight(features, labels, weights, learning_rate) :
	n = len(labels)
	predictions = predict(features, weights)
	gd = np.dot(features.T, (predictions - labels)) 
	gd = gd/n
	weights -= gd/n * learning_rate
	return weights

def train(features, labels, weights, learning_rate, iteration) :
	cost_his = []
	for i in range(iteration) : 
		weights = update_weight(features, labels, weights, learning_rate)
		cost = cost_function(features, labels, weights) 
		cost_his.append(cost)
	return weights, cost_his

x = np.array([2,3])

pygame.init()

screen = pygame.display.set_mode((1200, 700))

clock = pygame.time.Clock()


BACKGROUND = (50, 34, 117)
WHITE = (255, 255, 255)
PANEL = (191, 202, 230)
BLACK = (0, 0, 0)
RED = (223, 0, 41)
BLUE = (32, 90, 167)

font = pygame.font.SysFont('sans', 40)
font_small = pygame.font.SysFont('sans', 20)
text_class1 = font.render("Class 1", True, RED)
text_class2 = font.render("Class 2", True, BLUE)
text_finish = font.render("Finish sample data", True, BLACK)
running = True

points = []
labels = []
label = 0	
add = True

while running :
	clock.tick(60)
	screen.fill(BACKGROUND)
	mouse_x, mouse_y = pygame.mouse.get_pos()

	# Draw Interface
	# Draw panel
	pygame.draw.rect(screen, WHITE, (50, 100, 1100, 500))
	pygame.draw.rect(screen, PANEL, (55, 105, 1090, 490))

	# Class 1 button
	pygame.draw.rect(screen, WHITE, (100, 20, 200, 60))
	screen.blit(text_class1, (140, 25))

	# Class 2 button
	pygame.draw.rect(screen, WHITE, (400, 20, 200, 60))
	screen.blit(text_class2, (440, 25))

	# Finish sample data
	pygame.draw.rect(screen, WHITE, (100, 620, 300, 60))
	screen.blit(text_finish, (115, 625))

	# Draw position of mouse when mouse on panel :
	if 50 <= mouse_x <= 1150 and 100 <= mouse_y <= 600 :
		text_mouse = font_small.render("(" + str(mouse_x - 50) + " " + str(mouse_y - 100) + ")", True, BLACK)
		screen.blit(text_mouse, (mouse_x + 10, mouse_y))

	for event in pygame.event.get() :
		if event.type == pygame.QUIT :
			running = False
		if event.type == pygame.MOUSEBUTTONDOWN :
			# Creat points on panel
			if 50 <= mouse_x <= 1150 and 100 <= mouse_y <= 600 :
				if add == True : 
					point = [mouse_x - 50, mouse_y - 100]
					points.append(point)
					labels.append(label)
			

			# Change label to 1
			if 100 <= mouse_x <= 300 and 20 <= mouse_y <= 80 :
				label = 1

			if 400 <= mouse_x <= 600 and 20 <= mouse_y <= 80 :
				label = 0

			# Finish sample data
			if 100 <= mouse_x <= 400 and 620 <= mouse_y <= 680 :
				add = False
				points = np.array(points)
				labels = np.array(labels)
				labels = labels.reshape(labels.shape[0], 1)
				print(points)
				print(labels)
				weights, cost_his = train(points, labels, [[0.1], [0.2]], 0.001, 60)
				x = predict(np.array([0, 0]), weights)
				print(x)


	pygame.display.flip()

pygame.quit()