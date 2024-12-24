
---

## Building Ast Nodes

CodeGenerator module is implemented as a Builder software pattern

```mermaid
---
title:
    Builder
config:
    theme: base
    themeVariables:
        darkMode: true
        primaryColor: "#00ff00"
        padding: 50
    class:
        hideEmptyMembersBox: true
---

classDiagram
    CodeGenerator --> JaplFunction
    JaplFunction --> Call
    JaplFunction --> Proto
    JaplFunction --> Def

    class InvisibleNode {
        <<Hidden>>
    }

    Builder --|> FileBuilder : Inheritance

    class CodeGenerator
    CodeGenerator : +code_type
    CodeGenerator : +build_ext_module()
    CodeGenerator : +build_file()

    class Builder
    Builder : +data dict[filename, ast_node]
    Builder : +add_item()
    Builder : +do_build()

    class FileBuilder
    FileBuilder : +data dict[filename, ast_node]
    FileBuilder : +add_item()
    FileBuilder : +do_build()
```

---

## Ast

```mermaid
---
title:
config:
    theme: base
    themeVariables:
        darkMode: true
        primaryColor: "#00ff00"
        padding: 50
    class:
        hideEmptyMembersBox: true
---

flowchart LR

    subgraph Expr [Symbolic Expression]
        direction TB
    end

    subgraph Printable
        direction TB
        A --> B
    end

    subgraph Printer
        direction TB
        print_functions["_print_attribute()"]
    end

    subgraph Builder
        direction TB
    end

    subgraph Director
        direction TB
    end

    Expr --> Printable --> Printer --> Builder --> Director

```

    subgraph Main
        direction LR
        Expr(Symbolic Expression) --> Printable(Printable Types) --> Printer --> Builder(Builder) --> Director
    end
    Printable --> PrintableInner
    Printer --> PrinterInner
