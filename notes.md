## Notes

### Born-Digital dimensions to review before processing

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
- Remove 612x792 and 792x612 before processing
- Also seeing this phenomenon with 630x810 and 810x630
- uiext25291 and uiext25120 are duplicates on the live site

### crabtree

- Still vetting born-digital objects by dimension since attempts to automate are inaccurate
- This was the smoothest processing so far -- no issues with PDF files killing the script

### moscowcoop

- no interruptions but very slow -- in the later issues these are big, multi-columned documents

### blackhistory

- Odd, new phenomenon where the processing seems to be stripping archival scans of text: arg-02-10-1998-uiwsucelebrateblackhistory and others in the review folder. These are not born-digital.

### twrs and dworshak

- very quick and only one review item between both collections

