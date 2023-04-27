MODEL SOURCE:
- https://github.com/ozanciga/self-supervised-histopathology


ROADMAP:
- prune net so that output is (for CONIC data) [512, 7, 7] or [256, 16, 16].
    - or not so much? In the UNet paper the input images are (3, 512, 512) and the lowest feature maps are (1024, 30, 30).
- use net as encoder, with freezed layers, and build a specular decoder (to be trained from start)
    - the decoder should return the semantic segmentation mask (also the instance? Nope.)
    - add skip connections!
- train on CONIC (only pc and eos) (do hold-out to validate too)


Biomarkers:
- 2 = epithelium
- 4 = plasma cells
- 5 = eosinophils.


TODO MODEL:
- definisci classe encoder che carica rete pretrained (nn.Module), definisci classe decoder (nn.Module), infine definisci classe modello (pl.LightningModule) che si prendere l'encoder e il decoder.


PRUNING: n layers to remove from the end --> feature maps shape
- 1 --> 512, 1, 1
- 2 --> 512, 8, 8
- 3 --> 256, 16, 16 ***
- 4 --> 128, 32, 32
- 5 --> 64, 64, 64


CPATH-TRANSFER-LEARNING:
- Provare con pesi encoder pretrained vs pesi encoder inizializzati da zero.
- Opzioni: encoder freezato + decoder allenato, encoder fine-tuned + decoder allenato, encoder resettato e riallenato + decoder allenato.
- Prima fare decoder semplice (qualche layer e senza skip per provare funzionamento), poi fare decoder complesso.


Come gestire features maps di diversa shape derivanti da pruning diversi: input di qualsiasi dimensione, poi decoder raddoppia ad ogni passaggio ma mettendo un check in modo che quando raggiunge 256 si fermi.

TODO: definisci classe decoder.