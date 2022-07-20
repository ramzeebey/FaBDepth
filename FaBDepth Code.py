import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image, ImageFile
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True


def rescaleFrame(frame, scale=0.75):
    """Function to resize frames.

    Attributes:
        frame (numpy.ndarray): the frame to be resized.
        scale (float): the scale for resizing, by default, 75%.

    Returns:
        frame: the resized frame.
    """
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def capture_frames(video=False):
    """Function to resize frames.

    Attributes:
        video (str): video file can be passed or by default
        live video can be captured.
    """
    if video:
        capture = cv2.VideoCapture(video)
    else:  # live video
        capture = cv2.VideoCapture(0)

    i = 0  # counting frames

    while True:
        # capture frame-by-frame
        ret, frame = capture.read()

        # display live video
        cv2.imshow('video', frame)

        name = f'frame{i}.jpg'

        # save original frames
        

        # resize and save
        resized_frame = rescaleFrame(frame)
        cv2.imwrite(os.path.join(resized, name), resized_frame)

        i += 1  # incrementing count

        # video will run infinitely till 'q' is pressed on keyboard
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the capture and closing the video window after end
    capture.release()
    cv2.destroyAllWindows()


def decode_segmap(image, source, nc=21):
    # Apply the transformations needed
    # Define the helper function
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128,
                                                        0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64,
                                                              0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0,
                                                           128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)

    # Load the foreground input image
    foreground = cv2.imread(source)

    # Change the color of foreground image to RGB
    # and resize image to match shape of R-band in RGB output map
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    foreground = cv2.resize(foreground, (r.shape[1], r.shape[0]))

    # Create a background array to hold white pixels
    # with the same size as RGB output map
    background = 255 * np.zeros_like(rgb).astype(np.uint8)

    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Create a binary mask of the RGB output map using the threshold value 0
    th, alpha = cv2.threshold(np.array(rgb), 0, 255, cv2.THRESH_BINARY)

    # Apply a slight blur to the mask to soften edges
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background
    outImage = cv2.add(foreground, background)

    # Return a normalized output image for display
    return outImage/255


def segment(model, path, filename):
    input_image = Image.open(path)
    s_path = os.path.join(separated, filename)

    # Comment the Resize and CenterCrop for better inference results
    preprocess = T.Compose([T.Resize(450),
                            # T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

    input_batch = preprocess(input_image).unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]

    output_predictions = torch.argmax(
        output.squeeze(), dim=0).detach().cpu().numpy()

    rgb = decode_segmap(output_predictions, path)

    plt.imshow(rgb)
    plt.axis('off')
    plt.savefig(s_path)


def get_depth(path, filename):
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.eval()

    #Move model to GPU if available
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # Load transforms to resize and normalize the image for large or small model
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    transform = midas_transforms.default_transform

    # Load image and apply transforms
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

     #Predict and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    plt.imshow(output)
    plt.axis('off')
    plt.savefig(os.path.join(depth, filename)) ##


if __name__ == "__main__":
    # making folders
    current_directory = os.getcwd()
    current_directory = os.path.join(current_directory,'FaBDepth')

    # for saving the original images
    original = os.path.join(current_directory, 'original-images')
    os.makedirs(original)    

    # for saving the resized images
    resized = os.path.join(current_directory, 'resized-Gestures')
    os.makedirs(resized) 

    # for saving the fgbgs images
    separated = os.path.join(current_directory, 'FGBGS-Gestures')
    os.makedirs(separated)

    # for saving the depth images
    depth = os.path.join(current_directory, 'Depth-Gestures')
    os.makedirs(depth)

    # capturing the live video and saving frames
    capture_frames()

    dlab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

    num_of_frames = len(os.listdir(resized)) - 5

    # live capture fgbgs
    for i in range(5, num_of_frames, 10):
        path = ''
        filename = f'frame{i}.jpg'
        path = os.path.join(resized, filename)

        segment(dlab, path, filename)
        get_depth(path, filename)

        print(filename)
