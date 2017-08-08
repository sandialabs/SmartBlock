# SmartBlock
Reusable workflow components using ADIOS + FlexPath

Abstract:

Multi-step scientific workflows have become prominent and powerful tools of
data-driven scientific discovery. Run-time analytic techniques are now commonly
used to mitigate the performance effects of using parallel file systems as
staging areas during workflow execution. However, workflow construction and
deployment for extreme-scale computing is still largely an ad hoc process with
uneven support from existing tools. In this paper, we present SMARTBLOCK, an
approach to designing generic, reusable components for end-to-end construction
of workflows. Specifically, we demonstrate that a small set of SMARTBLOCK
generic components can be reused to build a diverse set of workflows, using
examples based on actual analytic processes with three well-known scientific
codes. Our evaluation shows promising scaling properties as well as negligible
overheads for using a modular approach over a custom, "all-in-one" solution. As
extreme-scale systems incorporate data analytics on simulation data as it is
generated at rates that far outstrip available I/O bandwidth, tools such as
SMARTBLOCK will become increasingly valuable for defining and deploying
flexible, efficient workflows.

This work has been published in IPDRM@IPDPS 2017:
https://doi.org/10.1109/IPDPSW.2017.149

This work was funded by a grant from the US Department of Energy, Office of Science under the guidance of Lucy Nowell for the Data Management grant program.
