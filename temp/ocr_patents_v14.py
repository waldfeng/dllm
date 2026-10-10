from pathlib import Path
import pymupdf
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from retrieve_patents_v14 import IDS

ocr = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)
root = Path('/tmp/opencode')
for number in IDS:
    if (root / f'US{number}A1_ocr.txt').exists():
        continue
    doc = pymupdf.open(root / f'US{number}A1.pdf')
    pages = sorted(set([0] + list(range(max(1,len(doc)-4),len(doc)))))
    texts = []
    for index in pages:
        pix = doc[index].get_pixmap(matrix=pymupdf.Matrix(1.8,1.8),alpha=False)
        image = np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,3)
        # Process columns separately to retain claim order.
        chunks = [image] if index == 0 else [image[:,:image.shape[1]//2],image[:,image.shape[1]//2:]]
        lines = []
        for chunk in chunks:
            result,_ = ocr(chunk)
            lines.extend(row[1] for row in (result or []))
        text = f'\nPAGE {index+1}\n' + '\n'.join(lines)
        texts.append(text)
    target = root / f'US{number}A1_ocr.txt'
    target.write_text('\n'.join(texts))
    print(number,'pages',pages,'characters',sum(map(len,texts)),flush=True)
