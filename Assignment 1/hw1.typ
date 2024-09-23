#set text(
  font: "New Computer Modern",
  size: 12pt
)

#set page(
  paper: "us-letter",
  margin: (x: 1in, y: 1in),
  header: context {
    [
      #context document.title
      #h(1fr)
      Rajeev Atla
      #line(length: 100%, stroke: 0.5pt)
    ]
  },
  footer: context {
    [
      #line(length: 100%, stroke: 0.5pt)
      #set align(center)
      — Page
      #counter(page).display(
        "1 of 1 —",
        both: true,
      )
    ]
  }
)

#set document(
  title: "Distributed Deep Learning HW 1",
  author: ("Rajeev Atla"),
  date: auto
)

#set par(
  justify: true
)

#let codeblock(filename) = raw(read(filename), block: true, lang: filename.split(".").at(-1))

#align(center, text(24pt)[
  *#context document.title*
])

#align(center, text(24pt)[
  Rajeev Atla
])

#let Cov(value) = $text("Cov")(#value)$

The Resnet18 model was trained on the CIFAR10 dataset for 20 epochs. 
The learning rate was set to 0.01,
and all other settings were left default.
An A100 GPU was used in NSERC Perlmutter's shared queue.
Please see the attached "resnet18.ipynb" for implementation details (PyTorch Lightning was used.)

Below is the trainng and validation accuracy over the 20 epochs.

#image("Accuracy.png")