## Notes

### CCC

- Very strange issue with the tool generating mostly blank documents that you need to troubleshoot. They reproduce some but not all of the images and no OCR. They are all recent docs and they are all 612x792.
    - cccidaho1223, cccidaho1238, cccidaho1239, cccidaho1801, cccidaho1879
- Attempting to resolve this through bypassing born-digital PDF files that already have OCR and may also contain tags that need to be preserved. The script now duplicates these identified born digital files in the B folder but does not process them.
- Removing cccidaho1838 -- a full page newspaper spread because it is breaking functionality due to size and I would like to keep opticolumn / opticolumns functionality distinct. That looks like the only non-clipping
- Also bumping on cccidaho1595.pdf -- possibly through RAM complications of the multiple models
