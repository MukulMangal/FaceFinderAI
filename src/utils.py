import cv2


def load_image(path):
    """
    Loads an image from the given path.
    """
    image = cv2.imread(path)

    if image is None:
        print(f"Error: Could not load image from '{path}'")
        return None

    print("Image loaded successfully!")
    print(f"Image Shape    : {image.shape}")
    print(f"Image Data Type: {image.dtype}")

    return image


def resize_image(image, target_width):
    """
    Resize an image while preserving its aspect ratio.

    Args:
        image (numpy.ndarray): Input image.
        target_width (int): Desired width.

    Returns:
        numpy.ndarray: Resized image.
    """

    # Get original dimensions
    h, w = image.shape[:2]

    # Calculate scale factor
    scale = target_width / w

    # Calculate new height
    new_height = int(h * scale)

    # Resize image
    resized = cv2.resize(image, (target_width, new_height))

    print("\nImage Resized Successfully!")
    print(f"Original Shape : {image.shape}")
    print(f"Resized Shape  : {resized.shape}")

    return resized


def draw_rectangle(image, x, y, w, h):
    """
    Draw a green rectangle on the image.

    Args:
        image: Input image
        x, y: Top-left corner
        w, h: Width and height

    Returns:
        Image with rectangle drawn
    """

    image_with_rectangle = image.copy()

    cv2.rectangle(
        image_with_rectangle,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),   # Green (BGR)
        2              # Thickness
    )

    return image_with_rectangle


def crop_region(image, x, y, w, h):
    """
    Crop a region using NumPy slicing.

    Args:
        image: Input image
        x, y: Top-left corner
        w, h: Width and height

    Returns:
        Cropped image
    """

    cropped = image[y:y+h, x:x+w]

    return cropped

def display_image(image, window_name="Image"):
    """
    Display an image in a window.
    """

    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(image, output_path):
    """
    Save an image to disk.
    """

    success = cv2.imwrite(output_path, image)

    if success:
        print(f"Image saved successfully: {output_path}")
    else:
        print("Failed to save image.")


def bgr_to_rgb(image):
    """
    Convert an image from BGR to RGB.

    Args:
        image (numpy.ndarray): Input BGR image.

    Returns:
        numpy.ndarray: RGB image.
    """

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return rgb_image

if __name__ == "__main__":

    image = load_image("dataset/wedding_images/test.jpg")

    if image is not None:

        resized = resize_image(image, 800)

        rgb_image = bgr_to_rgb(resized)

        print(f"BGR Shape : {resized.shape}")
        print(f"RGB Shape : {rgb_image.shape}")

        save_image(resized, "output/resized_bgr.jpg")
        save_image(rgb_image, "output/resized_rgb.jpg")

        # Manual bounding box
        x = 250
        y = 200
        w = 200
        h = 250

        # Draw rectangle
        boxed = draw_rectangle(resized, x, y, w, h)

        # Crop region
        cropped = crop_region(resized, x, y, w, h)

        print(f"Cropped Shape : {cropped.shape}")

        display_image(boxed, "Rectangle")

        display_image(cropped, "Cropped Region")

        save_image(boxed, "output/rectangle.jpg")

        save_image(cropped, "output/cropped.jpg")
