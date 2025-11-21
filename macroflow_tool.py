"""
RFdiffusion2 protein design tool for MacroFlow - LangChain 1.0 Standard.

This module implements the RFdiffusion2 protein backbone generation and design tool
using LangChain 1.0 best practices with dependency injection and Pydantic validation.
"""

import glob
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, validator

from macroflow_toolkit.common import ensure_directory, error_response
from macroflow_toolkit.decorators import validate_and_log
from macroflow_toolkit.deps import CommandRunner, PathResolver, WorkspaceBuilder

logger = logging.getLogger(__name__)


# ==================== Pydantic Models ====================


class RFdiffusionInput(BaseModel):
    """Input parameters for RFdiffusion2 protein design.

    This model defines and validates all input parameters for the rfdiffusion_design tool.
    """

    input_pdb: Optional[str] = Field(
        None,
        description="Path to input PDB file for scaffolding/motif-based design (optional for unconditional design)",
    )
    contigs: List[str] = Field(
        ...,
        description="Contig specification defining which parts to design vs. keep fixed (e.g., ['10-50,A163-181,30'])",
    )
    output_dir: Optional[str] = Field(
        None,
        description="Directory to save design results (auto-generated if not provided)",
    )
    num_designs: int = Field(
        10,
        ge=1,
        description="Number of designs to generate",
    )
    inference_steps: int = Field(
        50,
        ge=1,
        description="Number of diffusion inference steps (T)",
    )
    contig_atoms: Optional[str] = Field(
        None,
        description="Atomic-level motif specification (e.g., \"{'A518':'CG,OD1,OD2'}\")",
    )
    hotspot_residues: Optional[str] = Field(
        None,
        description="Hotspot residues to guide design (e.g., 'A1-4,B5,B8,B9')",
    )
    ligand: Optional[str] = Field(
        None,
        description="Ligand chain/residue name for small molecule binder design",
    )
    symmetry: Optional[str] = Field(
        None,
        description="Symmetry specification (e.g., 'C3', 'D2', 'I')",
    )
    contig_length: Optional[str] = Field(
        None,
        description="Total length constraint (e.g., '100-150')",
    )
    design_startnum: int = Field(
        0,
        ge=0,
        description="Starting index for design numbering",
    )
    deterministic: bool = Field(
        False,
        description="Use deterministic sampling for reproducibility",
    )

    @validator("contigs")
    def validate_contigs(cls, v):
        """Validate contig specification."""
        if not v or len(v) == 0:
            raise ValueError("At least one contig specification is required")
        return v


class RFdiffusionOutput(BaseModel):
    """Output from RFdiffusion2 protein design."""

    success: bool
    design_files: Optional[List[str]] = None
    trb_files: Optional[List[str]] = None
    output_dir: Optional[str] = None
    num_designs_generated: Optional[int] = None
    best_design: Optional[str] = None
    best_trb: Optional[str] = None
    lddt_scores: Optional[List[float]] = None
    mean_lddt: Optional[float] = None
    command: Optional[str] = None
    processing_time: Optional[float] = None
    input_pdb: Optional[str] = None
    contigs: Optional[List[str]] = None
    error: Optional[str] = None
    message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


# ==================== Helper Functions ====================


def _get_rfdiffusion_dir() -> Path:
    """Get the path to the RFdiffusion2 tool directory."""
    return Path(__file__).parent.resolve()


def _build_rfdiffusion_command(
    args: RFdiffusionInput,
    output_prefix: str,
) -> List[str]:
    """Build RFdiffusion2 inference command.

    Args:
        args: Validated input parameters
        output_prefix: Output file prefix path

    Returns:
        Command list for subprocess execution
    """
    cmd = [
        "python",
        "rf_diffusion/run_inference.py",
        "--config-name=base",
    ]

    # Input PDB
    if args.input_pdb:
        cmd.append(f"inference.input_pdb={args.input_pdb}")

    # Output configuration
    cmd.append(f"inference.output_prefix={output_prefix}")
    cmd.append(f"inference.num_designs={args.num_designs}")
    cmd.append(f"inference.design_startnum={args.design_startnum}")

    # Diffusion parameters
    cmd.append(f"diffuser.T={args.inference_steps}")

    # Deterministic mode
    if args.deterministic:
        cmd.append("inference.deterministic=True")

    # Contig specification
    contigs_str = "[" + ",".join([f"'{c}'" for c in args.contigs]) + "]"
    cmd.append(f"contigmap.contigs={contigs_str}")

    # Atomic-level motif specification
    if args.contig_atoms:
        cmd.append(f'contigmap.contig_atoms="{args.contig_atoms}"')

    # Length constraint
    if args.contig_length:
        cmd.append(f"contigmap.length={args.contig_length}")

    # Hotspot residues
    if args.hotspot_residues:
        cmd.append(f"inference.only_guidepost_positions={args.hotspot_residues}")

    # Ligand specification
    if args.ligand:
        cmd.append(f"inference.ligand={args.ligand}")

    # Symmetry
    if args.symmetry:
        cmd.append(f"sym.symid={args.symmetry}")

    return cmd


def _parse_rfdiffusion_output(
    output_prefix: str,
    num_designs: int,
    design_startnum: int,
    path_resolver: PathResolver,
) -> Dict:
    """Parse RFdiffusion2 output directory.

    Args:
        output_prefix: Output file prefix used in the command
        num_designs: Number of designs requested (unused, kept for API consistency)
        design_startnum: Starting design number (unused, kept for API consistency)
        path_resolver: Path resolution implementation

    Returns:
        Dictionary with parsed results
    """
    output_path = Path(output_prefix).parent

    # Find generated PDB files
    design_pattern = f"{output_prefix}_*.pdb"
    design_files = sorted(glob.glob(design_pattern))

    if not design_files:
        return error_response(
            "No design files found",
            output_dir=str(output_path),
            pattern=design_pattern,
        )

    # Find corresponding .trb files (trajectory/metadata files)
    trb_files = []
    lddt_scores = []

    for design_file in design_files:
        trb_file = design_file.replace('.pdb', '.trb')
        if Path(trb_file).exists():
            trb_files.append(trb_file)

            # Try to extract LDDT score from trb file
            try:
                import numpy as np
                trb_data = np.load(trb_file, allow_pickle=True)
                if 'lddt' in trb_data:
                    lddt = float(trb_data['lddt'].mean())
                    lddt_scores.append(lddt)
                elif 'inpaint_lddt' in trb_data:
                    lddt = float(np.mean(trb_data['inpaint_lddt']))
                    lddt_scores.append(lddt)
            except Exception as e:
                logger.warning(f"Failed to parse trb file {trb_file}: {e}")

    # Determine best design (highest LDDT if available)
    best_design = design_files[0]
    best_trb = trb_files[0] if trb_files else None

    if lddt_scores:
        best_idx = lddt_scores.index(max(lddt_scores))
        best_design = design_files[best_idx]
        best_trb = trb_files[best_idx] if best_idx < len(trb_files) else None

    # Convert paths to virtual paths
    def _to_virt(p: Optional[str]) -> Optional[str]:
        if not p:
            return p
        return path_resolver.to_virtual(Path(p))

    result = {
        "success": True,
        "design_files": [_to_virt(f) for f in design_files],
        "trb_files": [_to_virt(f) for f in trb_files],
        "output_dir": _to_virt(str(output_path)),
        "num_designs_generated": len(design_files),
        "best_design": _to_virt(best_design),
        "best_trb": _to_virt(best_trb),
    }

    if lddt_scores:
        result["lddt_scores"] = lddt_scores
        result["mean_lddt"] = sum(lddt_scores) / len(lddt_scores)

    return result


# ==================== Core Business Logic ====================


@validate_and_log
def run_rfdiffusion_design(
    args: RFdiffusionInput,
    runner: CommandRunner,
    workspace: WorkspaceBuilder,
    path_resolver: PathResolver,
) -> RFdiffusionOutput:
    """Execute RFdiffusion2 protein design with dependency injection.

    This is the pure business logic function that performs the actual design.
    It receives all dependencies as parameters, making it fully testable.

    Args:
        args: Validated input parameters
        runner: Command execution implementation
        workspace: Workspace creation implementation
        path_resolver: Path resolution implementation

    Returns:
        RFdiffusionOutput with design results
    """
    start_time = time.time()

    # Resolve input PDB path if provided
    input_path = None
    if args.input_pdb:
        system_path = Path(args.input_pdb)
        if system_path.is_absolute() and system_path.exists():
            input_path = system_path.resolve()
        elif args.input_pdb.startswith("/") or path_resolver.get_workspace_root() is not None:
            input_path = path_resolver.resolve(args.input_pdb)
        else:
            input_path = Path(args.input_pdb).resolve()

        if not input_path.exists():
            return RFdiffusionOutput(
                success=False,
                error="file_not_found",
                message=f"Input PDB file not found: {args.input_pdb}",
            )

    # Determine output directory
    if args.output_dir is None:
        if input_path and "workspace" in str(input_path.parent):
            output_dir = str(input_path.parent)
        else:
            output_dir = str(workspace.create("rfdiffusion"))
    else:
        if args.output_dir.startswith("/") or path_resolver.get_workspace_root() is not None:
            output_dir = str(path_resolver.resolve(args.output_dir))
        else:
            output_dir = args.output_dir

    output_path = Path(output_dir).resolve()
    ensure_directory(output_path)

    # Create output prefix
    output_prefix = str(output_path / "design")

    # Build command
    cmd = _build_rfdiffusion_command(args, output_prefix)

    # Execute command with uv wrapper
    rfdiffusion_dir = _get_rfdiffusion_dir()
    if not (rfdiffusion_dir / "uv.lock").exists():
        logger.warning("uv.lock not found. Run 'make prepare-rfdiffusion' if needed.")

    full_cmd = ["uv", "run"] + cmd
    cmd_result = runner.run(full_cmd, cwd=rfdiffusion_dir)

    if not cmd_result["success"]:
        error_msg = "RFdiffusion2 command failed"
        if "not found" in cmd_result.get("error", ""):
            error_msg = "RFdiffusion2 not installed. Run 'make prepare-rfdiffusion' to set it up."
        logger.error(f"{error_msg}: {cmd_result.get('error')}")
        return RFdiffusionOutput(
            success=False,
            error=error_msg,
            command=" ".join(full_cmd),
            stdout=cmd_result.get("stdout", ""),
            stderr=cmd_result.get("stderr", ""),
        )

    # Parse results
    results = _parse_rfdiffusion_output(
        output_prefix, args.num_designs, args.design_startnum, path_resolver
    )

    processing_time = time.time() - start_time

    # Convert to output model
    output = RFdiffusionOutput(
        **results,
        command=" ".join(full_cmd),
        processing_time=processing_time,
        input_pdb=path_resolver.to_virtual(input_path) if input_path else None,
        contigs=args.contigs,
    )

    logger.info(f"RFdiffusion2 design completed in {processing_time:.2f}s")
    return output


# ==================== LangChain 1.0 Tool Factory ====================


def make_tool(
    runner: CommandRunner,
    workspace: WorkspaceBuilder,
    path_resolver: PathResolver,
):
    """Factory function to create a LangChain 1.0 compatible RFdiffusion2 tool.

    This factory creates a tool with injected dependencies, enabling
    easy testing and customization.

    Args:
        runner: Command execution implementation
        workspace: Workspace creation implementation
        path_resolver: Path resolution implementation

    Returns:
        LangChain BaseTool instance

    Example:
        >>> from macroflow_toolkit import MacroflowToolkit
        >>> toolkit = MacroflowToolkit()
        >>> rfdiffusion_tool = make_tool(
        ...     toolkit.runner,
        ...     toolkit.workspace,
        ...     toolkit.path_resolver
        ... )
        >>> result = rfdiffusion_tool.invoke({
        ...     "contigs": ["10-50,A163-181,30"],
        ...     "num_designs": 10
        ... })
    """

    @tool
    def rfdiffusion_design(
        contigs: List[str],
        input_pdb: Optional[str] = None,
        output_dir: Optional[str] = None,
        num_designs: int = 10,
        inference_steps: int = 50,
        contig_atoms: Optional[str] = None,
        hotspot_residues: Optional[str] = None,
        ligand: Optional[str] = None,
        symmetry: Optional[str] = None,
        contig_length: Optional[str] = None,
        design_startnum: int = 0,
        deterministic: bool = False,
    ) -> str:
        """Design protein backbones using RFdiffusion2 from contig specifications.

        RFdiffusion2 is a powerful protein backbone generation tool that can:
        - Generate unconditional protein backbones
        - Scaffold protein motifs (fixed regions)
        - Design small molecule binders
        - Create symmetric assemblies
        - Perform atomic-level active site design

        Args:
            contigs: Contig specification defining structure (e.g., ['10-50,A163-181,30']
                means 10-50 residues, then keep residues 163-181 from chain A, then 30 residues)
            input_pdb: Path to input PDB file for motif scaffolding (optional)
            output_dir: Directory to save results (auto-generated if not provided)
            num_designs: Number of designs to generate (default: 10)
            inference_steps: Number of diffusion steps, more = higher quality (default: 50)
            contig_atoms: Atomic-level motif specification for active site design
                (e.g., "{'A518':'CG,OD1,OD2','A616':'CG,OD1,OD2'}")
            hotspot_residues: Specific residues to guide design (e.g., 'A1-4,B5,B8,B9')
            ligand: Ligand chain/residue name for small molecule binder design
            symmetry: Symmetry specification (e.g., 'C3' for 3-fold, 'D2' for dihedral)
            contig_length: Total length constraint (e.g., '100-150')
            design_startnum: Starting index for design numbering (default: 0)
            deterministic: Use deterministic sampling for reproducibility (default: False)

        Returns:
            JSON string containing:
            - success: Whether design succeeded
            - design_files: List of generated PDB files
            - best_design: Path to best design (highest LDDT if available)
            - lddt_scores: Confidence scores for each design
            - processing_time: Time taken in seconds

        Examples:
            # Unconditional backbone generation
            {"contigs": ["100-150"]}

            # Motif scaffolding
            {"contigs": ["10-50,A163-181,30"], "input_pdb": "/workspace/motif.pdb"}

            # Small molecule binder
            {"contigs": ["150"], "input_pdb": "/workspace/ligand.pdb", "ligand": "LIG"}

            # Symmetric design
            {"contigs": ["50"], "symmetry": "C3"}
        """
        # Validate inputs
        try:
            args = RFdiffusionInput(
                input_pdb=input_pdb,
                contigs=contigs,
                output_dir=output_dir,
                num_designs=num_designs,
                inference_steps=inference_steps,
                contig_atoms=contig_atoms,
                hotspot_residues=hotspot_residues,
                ligand=ligand,
                symmetry=symmetry,
                contig_length=contig_length,
                design_startnum=design_startnum,
                deterministic=deterministic,
            )
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return json.dumps({"success": False, "error": f"Invalid input: {e}"})

        # Execute design
        result = run_rfdiffusion_design(args, runner, workspace, path_resolver)

        # Return JSON string for LLM consumption
        return result.model_dump_json(exclude_none=True, indent=2)

    # Store interrupt config and runtime metadata in tool's metadata field
    # This is the LangChain 1.0 standard way to attach custom data to tools
    if not hasattr(rfdiffusion_design, 'metadata') or rfdiffusion_design.metadata is None:
        rfdiffusion_design.metadata = {}

    rfdiffusion_design.metadata.update({
        "category": "protein_design",
        "interrupt_config": {
            "enabled": True,
            "basic_params": ["contigs", "input_pdb", "num_designs"],
            "advanced_params": ["inference_steps", "contig_atoms", "hotspot_residues", "ligand", "symmetry"],
            "message": "RFdiffusion2 蛋白质设计参数确认\n预计耗时: 5-30 分钟（取决于设计数量和推理步数）",
        },
        "requires_gpu": True,
        "avg_runtime": "5-30 minutes",
        "input_formats": ["pdb"],
        "output_format": "pdb",
        "advanced_tool": True,
    })

    return rfdiffusion_design
