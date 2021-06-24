import numpy as np
import vispy
import vispy.scene.visuals


def pcd_vispy(img=None, name=None, vis_size=(800, 600)):
    canvas = vispy.scene.SceneCanvas(title=name, keys='interactive', size=vis_size, show=True)
    canvas.measure_fps()
    canvas.show()

    view = canvas.central_widget.add_view()
    photo = vispy.scene.visuals.Image(data=img, parent=view.scene, method='auto', interpolation="catrom")
    # doc https://www.ssec.wisc.edu/~davidh/tmp/vispy/api/vispy.visuals.image.html#vispy.visuals.image.ImageVisual

    view.camera = 'panzoom'
    view.camera.flip = (False, True)
    view.camera.aspect = 1.0
    # view.camera.set_range((0, 800), (0, 600))
    view.camera.set_range()
    view.camera.fov = 0

    return photo, canvas


def update_image():
    image = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    photo.set_data(image)
    photo.update()
    canvas.update()


def on_timer(event):
    update_image()


WIDTH = 1920
HEIGHT = 1200


if __name__ == '__main__':
    image = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
    photo, canvas = pcd_vispy(img=image)

    timer = vispy.app.Timer(interval='auto', connect=on_timer)
    timer.start()

    vispy.app.run()
