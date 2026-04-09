## Notes

### CCC

- Very strange issue with the tool generating mostly blank documents that you need to troubleshoot. They reproduce some but not all of the images and no OCR. They are all recent docs and they are all 612x792.
    - cccidaho1223, cccidaho1238, cccidaho1239, cccidaho1801, cccidaho1879
- Attempting to resolve this through bypassing born-digital PDF files that already have OCR and may also contain tags that need to be preserved. The script now duplicates these identified born digital files in the B folder but does not process them.
- Removing cccidaho1838 -- a full page newspaper spread because it is breaking functionality due to size and I would like to keep opticolumn / opticolumns functionality distinct. That looks like the only non-clipping
- Also bumping on cccidaho1595.pdf -- possibly through RAM complications of the multiple models

### Harvester

- Removing all of the viewscan items to run on opticolumns later
- Might want to process this one again

#### Iterative Note: Born-Digital Items

- I may have misidentified the problem with the 612x792 files. I wanted to have the tool identify docs that have already been tagged but this appear to be a complicated thing to identify. This shouldn't come up too often to incorporate into the script but it might be worth creating a new iteration of the script for more contemporary collections. For now, I just reverted to an earlier version and added preprocessing that looks for items that have already been generated in the B folder and then skips them. 

### uiext

- A little back and forth while fine tuning the script.
