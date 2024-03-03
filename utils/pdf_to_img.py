import os
from pdf2image import convert_from_path
import cv2


def pdf_to_img(pdf_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pages = convert_from_path(pdf_file)
    for i, page in enumerate(pages):
        filename = os.path.join(output_dir, f"page_{i+1}.jpg")
        page.save(filename, 'JPEG')
    print("Images saved in the directory:", output_dir)

def cut_half(image_path):
    img = cv2.imread(image_path)
    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2
    s1 = img[:, :width_cutoff]
    s2 = img[:, width_cutoff:]
    return s1, s2

if __name__ == "__main__":
    pdf_file = "pdf/Padilla Nobleza virtuosa test.pdf"

    if not os.path.exists('images'):
        os.makedirs('images')

    pdf_to_img(pdf_file,"images")
    
    lis_images = os.listdir('images')

    lis_images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))


    if not os.path.exists('pages'):
        os.makedirs('pages')

    for image in range(len(lis_images)):
        s1, s2 = cut_half('images/'+lis_images[image])
        cv2.imwrite(f'pages/{image*2}.jpg', s1)
        cv2.imwrite(f'pages/{(image*2)+1}.jpg', s2)