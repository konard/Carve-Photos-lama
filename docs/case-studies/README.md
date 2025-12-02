# LaMa Case Studies

This directory contains in-depth case studies and technical analyses of key decisions and issues in the LaMa project.

## Available Case Studies

### 1. torch.dynamo_export Limitations (Issue #10)

**Directory**: [torch-dynamo-export-issue/](torch-dynamo-export-issue/)

**Summary**: Comprehensive analysis of why `torch.dynamo_export` is not suitable for LaMa ONNX export, documenting the investigation process, findings, and decision to use traditional export with custom FFT implementation.

**Key Documents**:
- [Main Case Study](torch-dynamo-export-issue/README.md) - Complete analysis and recommendations
- [Timeline](torch-dynamo-export-issue/timeline.md) - Chronological sequence of events
- [Root Cause Analysis](torch-dynamo-export-issue/root-cause-analysis.md) - Deep technical analysis
- [Solutions & Recommendations](torch-dynamo-export-issue/solutions-and-recommendations.md) - Actionable guidance
- [Raw Data](torch-dynamo-export-issue/raw-data/) - Archived source material

**Status**: ✅ Documented

**Related Issues**:
- [Carve-Photos/lama#10](https://github.com/Carve-Photos/lama/issues/10)
- [advimman/lama#315](https://github.com/advimman/lama/issues/315)
- [pytorch/pytorch#107588](https://github.com/pytorch/pytorch/issues/107588)
- [pytorch/pytorch#133785](https://github.com/pytorch/pytorch/issues/133785)
- [pytorch/pytorch#125903](https://github.com/pytorch/pytorch/issues/125903)

---

## Purpose of Case Studies

Case studies in this directory serve to:

1. **Document Decision-Making**: Preserve the reasoning behind technical choices
2. **Share Knowledge**: Help the community avoid repeating the same investigations
3. **Provide Context**: Explain why certain approaches were chosen over alternatives
4. **Enable Future Re-evaluation**: Create triggers for when to revisit decisions
5. **Maintain Institutional Memory**: Ensure knowledge persists as team members change

## How to Use These Case Studies

### For Users
If you're wondering why LaMa uses a particular approach (like custom FFT for ONNX export), check the relevant case study for the full context and reasoning.

### For Contributors
Before proposing changes to established patterns, review case studies to understand:
- What was tried before
- Why certain approaches were rejected
- What conditions would justify changing the current approach

### For Researchers
Case studies provide real-world examples of:
- Production ML system trade-offs
- Framework limitations and workarounds
- Decision-making under uncertainty

## Contributing New Case Studies

When documenting a new case study:

1. **Create a directory**: `docs/case-studies/descriptive-name/`
2. **Include core documents**:
   - `README.md` - Main case study document
   - `timeline.md` - Chronological events
   - `root-cause-analysis.md` - Technical deep dive
   - `solutions-and-recommendations.md` - Actionable guidance
   - `raw-data/` - Source material and evidence

3. **Follow the template structure** (see torch-dynamo-export-issue as example)
4. **Update this index** with the new case study
5. **Link from related code** using comments

### Case Study Template

```markdown
# Case Study: [Title]

## Executive Summary
- What was investigated
- Key findings
- Final decision

## Background
- Context and motivation
- Problem statement

## Investigation Process
- What was tested
- Methods used

## Findings
- Detailed results
- Evidence

## Root Cause Analysis
- Why issues occurred
- Technical mechanisms

## Decision
- Final recommendation
- Implementation details
- When to revisit

## Timeline
- Chronological events

## References
- Sources and links

## Appendices
- Supporting material
```

---

## Maintenance

Case studies should be updated when:
- New evidence becomes available
- Underlying technologies change significantly
- Decisions are re-evaluated or changed
- Community reports new findings

**Maintainer**: Carve.Photos Team / Community \
**Last Updated**: 2025-12-02
