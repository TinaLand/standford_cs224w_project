# Hardware Requirements

## Minimum Requirements

### CPU
- **Processor**: 4+ cores recommended
- **Architecture**: x86_64 or ARM64 (Apple Silicon)
- **Performance**: Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)

### Memory (RAM)
- **Minimum**: 8 GB
- **Recommended**: 16 GB or more
- **For Full Pipeline**: 16 GB recommended (handles large graph datasets)

### Storage
- **Minimum**: 10 GB free space
- **Recommended**: 20 GB+ free space
- **Breakdown**:
  - Raw data: ~500 MB
  - Processed features: ~1 GB
  - Graph snapshots: ~2-3 GB (2,317 graphs)
  - Trained models: ~500 MB - 2 GB
  - Results and logs: ~500 MB

### GPU (Optional but Recommended)
- **For Training**: NVIDIA GPU with CUDA support (6GB+ VRAM recommended)
- **Supported**: CUDA 11.0+ or ROCm (AMD)
- **Without GPU**: Training will use CPU (significantly slower, ~10x slower)

## Recommended Setup

### Development/Testing
- **CPU**: 8+ cores
- **RAM**: 16 GB
- **Storage**: 20 GB SSD
- **GPU**: Optional (CUDA-capable GPU with 6GB+ VRAM)

### Production/Full Pipeline
- **CPU**: 12+ cores
- **RAM**: 32 GB
- **Storage**: 50 GB+ SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/3080 or better)

## Performance Expectations

### With GPU (NVIDIA RTX 3080 / 10GB VRAM)
- **Phase 1** (Data Collection): ~5-10 minutes
- **Phase 2** (Graph Construction): ~15-30 minutes
- **Phase 3** (Baseline Training): ~30-60 minutes
- **Phase 4** (Transformer Training): ~1-2 hours
- **Phase 5** (RL Training): ~30-60 minutes
- **Phase 6** (Evaluation): ~10-20 minutes
- **Total**: ~3-5 hours

### Without GPU (CPU Only)
- **Phase 1** (Data Collection): ~5-10 minutes
- **Phase 2** (Graph Construction): ~15-30 minutes
- **Phase 3** (Baseline Training): ~3-6 hours
- **Phase 4** (Transformer Training): ~8-15 hours
- **Phase 5** (RL Training): ~2-4 hours
- **Phase 6** (Evaluation): ~30-60 minutes
- **Total**: ~15-30 hours

## Software Requirements

### Operating System
- **Linux**: Ubuntu 20.04+ or similar
- **macOS**: 10.15+ (Catalina or later)
- **Windows**: Windows 10+ (WSL2 recommended)

### Python
- **Version**: Python 3.8 - 3.11
- **Recommended**: Python 3.9 or 3.10

### Key Dependencies
- PyTorch 1.12+ (with CUDA if using GPU)
- PyTorch Geometric 2.0+
- NumPy, Pandas, Scikit-learn
- See `requirements.txt` for complete list

## Memory Usage by Phase

1. **Phase 1** (Data Collection): ~2-4 GB peak
2. **Phase 2** (Graph Construction): ~4-8 GB peak
3. **Phase 3** (Baseline Training): ~6-12 GB peak (with GPU: ~2-4 GB VRAM)
4. **Phase 4** (Transformer Training): ~8-16 GB peak (with GPU: ~4-8 GB VRAM)
5. **Phase 5** (RL Training): ~4-8 GB peak (with GPU: ~2-4 GB VRAM)
6. **Phase 6** (Evaluation): ~4-8 GB peak

## Tips for Limited Resources

1. **Reduce Number of Stocks**: Modify `NUM_STOCKS` in data collection (default: 50)
2. **Reduce Date Range**: Use shorter time periods for testing
3. **Use CPU**: Works but much slower (no GPU required)
4. **Batch Processing**: Process graphs in smaller batches
5. **Clear Cache**: Delete intermediate files between phases

## Cloud Options

### Google Colab (Free Tier)
- **GPU**: Tesla T4 (16GB VRAM) - Free tier limited hours
- **RAM**: ~12-15 GB
- **Storage**: ~100 GB
- **Suitable for**: Testing and small experiments

### AWS / GCP / Azure
- **Recommended Instance**: 
  - GPU: g4dn.xlarge (NVIDIA T4, 16GB VRAM)
  - CPU: 4 vCPU, 16GB RAM
- **Cost**: ~$0.50-1.00/hour
- **Suitable for**: Full pipeline runs

## Troubleshooting

### Out of Memory Errors
- Reduce batch size in training scripts
- Process data in smaller chunks
- Use gradient checkpointing (already enabled in some models)
- Close other applications

### Slow Training
- Enable GPU if available
- Reduce model size (hidden_dim, num_layers)
- Use mixed precision training (AMP) - already enabled
- Reduce number of training epochs for testing

### CUDA Out of Memory
- Reduce batch size
- Use gradient accumulation
- Enable gradient checkpointing
- Reduce model dimensions

