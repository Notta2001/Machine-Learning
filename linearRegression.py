import pygame
import math
import numpy as np

pygame.init()

def normalEquation(list_x, list_y) :
	# Create vector one
	ones = np.ones((list_x.shape[0], 1), dtype = np.int8)

	#Combine 1 and list_x 
	list_x = np.concatenate((ones, list_x), axis = 1)

	return np.linalg.inv(list_x.transpose().dot(list_x)).dot(list_x.transpose()).dot(list_y)



screen = pygame.display.set_mode((1200,700))
pygame.display.set_caption("Linear Regression By Thang Doan")
running = True
clock = pygame.time.Clock()

BACKGROUND = (0, 178, 191)
WHITE = (255, 255, 255)
PANEL = (202, 229, 233)
BLACK = (0, 0, 0)
POINT = (0, 174, 114)
PINK = (197, 124, 172)

font = pygame.font.SysFont('sans', 40)
font_small = pygame.font.SysFont('sans', 20)
text_normalEquation = font.render("Normal Equation", True, BLACK)
text_reset = font.render("Reset", True, BLACK)
points = []
normal = 0
ERROR = 0

while running :
	clock.tick(60)
	screen.fill(BACKGROUND)
	mouse_x, mouse_y = pygame.mouse.get_pos()
	

	# Draw interface 
	# Draw panel 
	
	pygame.draw.rect(screen, WHITE, (50, 100, 1100, 500))
	pygame.draw.rect(screen, PANEL, (55, 105, 1090, 490))

	# Normal Equation button
	
	pygame.draw.rect(screen, PANEL, (400, 620, 300, 60))
	screen.blit(text_normalEquation, (430, 625))

	# ERROR 
	text_ERROR = font.render("ERROR : " + str(ERROR), True, BLACK)
	screen.blit(text_ERROR,(100, 20))

	# Reset button
	pygame.draw.rect(screen, PANEL, (800, 20, 300, 60))
	screen.blit(text_reset, (900,25))

	# Draw position of point when mouse in panel
	if 50 < mouse_x < 1150 and 100 < mouse_y < 600 :
		text_mouse = font_small.render("(" + str(mouse_x-50) + ", " + str(mouse_y-100) + ")", True, BLACK)
		screen.blit(text_mouse, (mouse_x+10, mouse_y))

	for event in pygame.event.get() :
		if event.type == pygame.QUIT :
			running = False
		if event.type == pygame.MOUSEBUTTONDOWN : 
			#Create point on panel
			if 50 < mouse_x < 1150 and 100 < mouse_y < 600 :
				point = [mouse_x - 50, mouse_y - 100]
				points.append(point)

			if 400 < mouse_x < 700 and 620 < mouse_y < 680 :
				print("Normal Equation")
				points = np.array(points)
				x = points[:,0]
				y = points[:,1]
				x = x.reshape(points.shape[0], 1)
				y = y.reshape(points.shape[0], 1)
				w = normalEquation(x, y)
				print(w)
				x_normal = np.array([min(x) - 10, max(x) + 10]).T
				y_normal = w[0][0] + x_normal*w[1][0]
				print(y_normal)
				normal = 1
				print(y[0])


				for i in range(len(x)) :
					y_cost = w[0][0] + x[i]*w[1][0]
					print(y_cost)
					ERROR += int((y_cost[0] - y[0][0])**2)

			if 800 < mouse_x < 1100 and 20 < mouse_y < 80 :                           
				points = []
				normal = 0
				ERROR = 0
	# print points 
	for point in points : 
		pygame.draw.circle(screen, POINT, (point[0] + 50, point[1] + 100), 6)

	# Draw line from normal equation
	if normal == 1 :
		pass
		pygame.draw.line(screen, PINK, [x_normal[0][0]+ 40, y_normal[0][0] + 100], [x_normal[0][1] + 60, y_normal[0][1] + 100], 5)

	pygame.display.flip() 

pygame.quit()