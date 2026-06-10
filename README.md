# opticolumn

OCR tool developed by Andrew Weymouth, Digital Initiatives Librarian for University of Idaho, over summer and fall of 2025. The tool implements the TrOCR text recognition model and the Kraken BLLA page segmentation model to improve the accuracy of handwritten and cursive archival documents and add digital preservation metadata to processed materials. The tool was developed for overhauling the Center for Digital Inquiry and Learning's digital collection PDF files, to make the collection more discoverable and accessible. The development of the tool is written about in greater detail in _Transparent Practices: OCR and AI in the Archives_, by Rebecca Hastings and Andrew Weymouth. Forthcoming in _Collections: A Journal for Archives and Museum Professions_, 2026.

## Note

Opticolumn and Opticolumns are designed for archival scanned documents, not born-digital PDF files. The tool creates a new OCR layer for scans that are missing OCR or replaces existing OCR when necessary, but it does not create a tagging structure or add other accessibility features.

If Opticolumn encounters a born-digital PDF, the issue will be obvious. The processing will strip the file of its text and images, leaving a largely blank document. These files can then be easily identified in the B folder and removed while retaining the original PDFs.

Ideally, the tool would automatically detect and skip born-digital files during processing. However, the wide variation in the formatting and metadata of born-digital PDFs has produced too many false positives. As a result, users should review their batches manually. 

Before processing, check for page dimensions commonly associated with born-digital PDFs. Keep in mind that these dimensions are only indicators and do not guarantee that a file is born digital.

- 396x612
- 540x960
- 612x792
- 630x810
- 720x540
- 756x972 
- 792x612 
- 810x630
- 841x1190 
- 960x540
- 960x720
- 1024x768 
- 1190x841 
- 1224x792 
- 1280x720 
- 1920x1080 

After processing, review the output files in the B folder. Born-digital PDFs will typically appear largely blank. Remove these files before updating or publishing the processed batch.
