MODEL SOURCE:
- https://github.com/ozanciga/self-supervised-histopathology


Biomarkers:
- 2 = epithelium
- 4 = plasma cells
- 5 = eosinophils.


PRUNING: n layers to remove from the end --> feature maps shape
- 1 --> 512, 1, 1
- 2 --> 512, 8, 8
- 3 --> 256, 16, 16 ***
- 4 --> 128, 32, 32 **
- 5 --> 64, 64, 64