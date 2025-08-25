---
title: Text in canvas not centered
source: https://www.reddit.com/r/ObsidianMD/comments/179un8s/text_in_canvas_not_centered/?show=original
author:
  - RasePL
published: 2023-10-17
created: 2025-08-24
description:
tags:
  - clippings
---
Hello, im trying to center text on canvas. Using code below it center text, but not exactly. There's always more space on right than on left

---

## 1 Comments

> **Siradal** • [18 points](https://reddit.com/r/ObsidianMD/comments/179un8s/comment/l8ulmgq/) •
> 
> For those that might not understand, here is a step by step instruction to help you easy understand what [No-Marionberry5313](https://www.reddit.com/user/No-Marionberry5313/) and then [Budget\_Impressive](https://www.reddit.com/user/Budget_Impressive/) refined:
> 
> 1. Go to **Obsidian settings**
> 2. In the window that pops-up go to **Appearance** and scroll down till you see **CSS snippets**.
> 3. You should see a folder to the right that lets you open the snippets folder. click on it
> 4. Your windows folder should pop-up and there you right click on an empty field and choose **New** -> **Textdocument**
> 5. Rename the textdocument to Canvas-Snippets.css (confirm your want to rename it)
> 6. Double click this file and then copy the css text Budget\_Impressive wrote down and save the file (close and confirm to save)
> 7. Close the window folder and don't forget to activate the **Canvas-Snippets** (your Obsidan setting should still be open)
> 
> Thx for your help guys!
> 
> track me
> 
> > **\[deleted\]** • [1 points](https://reddit.com/r/ObsidianMD/comments/179un8s/comment/lbgfcm7/) •
> > 
> > Worked perfectly for me, ty homie!
> > 
> > **Friendly\_Carpet8316** • [1 points](https://reddit.com/r/ObsidianMD/comments/179un8s/comment/lj777h0/) •
> > 
> > Thank you for these steps!!! I couldn't have done this without your help!
> > 
> > **RinoTT** • [1 points](https://reddit.com/r/ObsidianMD/comments/179un8s/comment/llakroc/) •
> > 
> > Thank You
> > 
> > **MrOddin** • [1 points](https://reddit.com/r/ObsidianMD/comments/179un8s/comment/lpqhpd2/) •
> > 
> > Thank you boss!
> > 
> > **UnderstandingNo807** • [1 points](https://reddit.com/r/ObsidianMD/comments/179un8s/comment/lzmjn5t/) •
> > 
> > Бро, ты лучший !!! Спасибо тебе

> **AStarNamedAltair** • [2 points](https://reddit.com/r/ObsidianMD/comments/179un8s/comment/n29e9zf/) •
> 
> Should anyone need it; I made a modified version of the code suggested that only centers headings (H1, H2, H3...), but allows for lists AND body paragraphs to remain left justified (not full justified.) [I needed it set up like that for longer definitions/descriptions of steps within a flow chart.](https://imgur.com/a/hK4abwB)
> 
> /\* Centers Headings \*/
> .canvas-node-content h1,
> .canvas-node-content h2,
> .canvas-node-content h3,
> .canvas-node-content h4,
> .canvas-node-content h5,
> .canvas-node-content h6 {
>     text-align: center;
> }
> 
> /\* Left-Aligns Body Paragraphs and Lists \*/
> .canvas-node-content p,
> .canvas-node-content li {
>     text-align: left;
> }