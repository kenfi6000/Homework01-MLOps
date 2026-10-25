# Fashion-MNIST sample images

This directory contains five 28x28 grayscale PNG images for each of the ten
Fashion-MNIST classes. The images are samples from the official test split.
Filenames start with the numeric class ID and class name; for example,
`6_shirt_01.png` is a shirt.

These samples were selected because the current model predicts their labeled
class correctly. They are convenient for smoke-testing the upload and
prediction flow, but are not an unbiased evaluation set.

Class IDs:

| ID | Label |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Upload an image in Streamlit at `http://127.0.0.1:8501/`, or test the API from
the repository root in PowerShell:

```powershell
curl.exe -F "file=@tests\fashion_mnist_samples\6_shirt_01.png;type=image/png" http://127.0.0.1:8000/predict
```
