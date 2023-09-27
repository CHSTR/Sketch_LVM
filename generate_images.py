import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from experiments.options import opts


save_results = 'save_images/'
if not os.path.exists(save_results):
    os.makedirs(save_results)

path_file = 'results/' + opts.output_file + '_images.txt'



# Lee el archivo de texto y almacena las rutas en una lista
image_paths = []
with open(path_file, "r") as file:
    for line in file:
        columns = line.strip().split(", ")  # Suponiendo que las columnas están separadas por ", "
        labels = columns[0]  # Etiqueta de la imagen de consulta
        query_image_path = columns[1]  # Ruta de la imagen de consulta
        result_image_paths = columns[2:]  # Rutas de las 10 imágenes de resultados
        image_paths.append((query_image_path, result_image_paths))


def plot_best_10_multiple(image_paths, name, lines_width=8):
    fig, axes = plt.subplots(nrows=len(image_paths), ncols=11, figsize=(int(11*1.5), int(len(image_paths)*1.5)))

    for i, (query, images) in enumerate(image_paths):
        selected = Image.open(query)
        selected = right_border_image(selected, lines_width)
        axes[i][0].imshow(selected)
        axes[i][0].axis('off')
        axes[i][1].axis('off')

        for j, image_path in enumerate(images):
            image = Image.open(image_path)
            axes[i][j+1].imshow(image)
            axes[i][j+1].axis('off')

    # Save figure
    plt.savefig(save_results + name + ".png", bbox_inches='tight', pad_inches=0)
    
    return fig, axes

def right_border_image(image, border_width):
    width, height = image.size
    new_width = width + border_width
    new_image = Image.new("RGB", (new_width, height))
    new_image.paste(image, (0, 0))
    return new_image

print("Generando imágenes")
for i in tqdm(range(0, len(image_paths), 5)):
    fig, axes = plot_best_10_multiple(image_paths[i:i+5], name="multiple_images_" + str(i))

