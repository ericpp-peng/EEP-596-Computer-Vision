// =======================================================
// EE P 596  Computer Vision: Classical and Deep Methods task_outputs Report Template (Typst)
// Author: Po Peng (Eric)
// =======================================================

// --- Metadata ---
#let course = "EEP 596A 
Computer Vision: Classical and Deep Methods"
#let quarter = "2025 Fall"
#let hw_title = "Homework 4 Report"
#let name = "Po Peng"
#let netid = "ericpp"

// --- Global Styles ---
#set page(width: 8.5in, height: 11in, margin: 1in)
#set text(size: 11pt)
#show heading: set text(14pt, weight: "bold")
#show figure.caption: set text(10pt, gray)
#show "```": block.with(fill: luma(98%), inset: 6pt, radius: 4pt, stroke: luma(90%))

#let today = datetime.today()


// --- Cover Page ---
#align(center)[
  #v(2em)
  #set text(size: 20pt, weight: "bold")
  #(course)
  #v(-0.5em)
  #set text(size: 14pt)
  #(quarter)
  #v(2em)
  #set text(size: 20pt, weight: "bold")
  #(hw_title)
  #v(2em)
  #set text(size: 16pt)
  *Name:* #name \
  *NetID:* #netid \
  *Date:* #today.display("[month repr:long] [day], [year]") \
  #v(6em)
  #set text(size: 11pt)
]

// --- Table of Contents ---
#v(0.5em)
Contents
#v(0.2em)
#outline(title: [], depth: 2)

// --- Tasks ---
#v(1em)
#v(1em)
#v(1em)
#v(1em)
#v(1em)
#v(1em)
#v(1em)
#v(1em)

= Task 1 – CIFAR-10 dataset

```
num_train_batches: 5
num_test_batches: 1
num_img_per_batch: 10,000
num_train_img: 50,000
num_test_img: 10,000
size_batch_bytes: 30,730 KB
size_image_bytes: 3.072 KB
size_batchimage_bytes: 30,720 KB
```
#figure(
  image("CIFAR10_batch4_visualization.png", width: 7in),
)

= Task 2 – Train classifier
```
(base) eric@ericdeMacBook-Pro homework4 % python Assignment4.py
[1,  2000] loss: 2.221
[1,  4000] loss: 1.917
[1,  6000] loss: 1.684
[1,  8000] loss: 1.582
[1, 10000] loss: 1.530
[1, 12000] loss: 1.464
[2,  2000] loss: 1.399
[2,  4000] loss: 1.372
[2,  6000] loss: 1.353
[2,  8000] loss: 1.355
[2, 10000] loss: 1.283
[2, 12000] loss: 1.279
Accuracy of the network on the 10000 test images: 54.7 %
```