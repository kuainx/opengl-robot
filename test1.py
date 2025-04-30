import glfw
from OpenGL.GL import *

glfw.init()
window = glfw.create_window(800, 600, "glfw first", None, None)
glfw.make_context_current(window)

while not glfw.window_should_close(window):
    glClearColor(0.2, 0.3, 0.3, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_TRIANGLES)
    glVertex3f(-0.5, -0.5, 0.0)
    glVertex3f(0.5, -0.5, 0.0)
    glVertex3f(0.0, 0.5, 0.0)
    glEnd()

    glfw.swap_buffers(window)
    glfw.poll_events()
