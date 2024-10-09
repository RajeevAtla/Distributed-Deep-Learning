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
  title: "Distributed Deep Learning HW 2",
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


In this assignment, we trained 2 different versions of Resnet18 on the CIFAR10 dataset.
A node of 4 Nvidia A100 GPUs were used on NSERC's Perlmutter supercomputer for the training.
Please see the attached Python files for training details.
(Pytorch Lightning's DDP strategy was used for ease of usage.)

The first version used data parallelism with 4 GPUs.
We can plot the validation accuracy vs time.

#image("val_acc_data_parallel.png")

The second version used model parallelism with 2 GPUs.
The model was split up into 2 parts
\- 1 with 2.8 million parameters,
and 1 with 8.4 million parameters.

#image("val_acc_model_parallel.png")