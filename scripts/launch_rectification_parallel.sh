#!/bin/bash
# Launch 4 parallel rectification jobs using SLURM job arrays

echo "================================================"
echo "   🚀 LAUNCHING PARALLEL RECTIFICATION"
echo "================================================"
echo ""
echo "Configuration:"
echo "  • Jobs: 4 parallel"
echo "  • CPU per job: 4"
echo "  • RAM per job: 8 GB"
echo "  • Total resources: 16 CPU, 32 GB RAM"
echo "  • Panoramas: 5,110 total (≈1,278 per job)"
echo ""

# Submit array job
JOB_ID=$(sbatch scripts/slurm_rectification_parallel.sh 2>&1 | grep -oP 'Submitted batch job \K\d+')

if [ -z "$JOB_ID" ]; then
    echo "❌ Failed to submit jobs!"
    exit 1
fi

echo "✅ Submitted job array: $JOB_ID"
echo ""
echo "Job IDs created:"
echo "  • ${JOB_ID}_0 → Batch 0: panoramas 0-1277"
echo "  • ${JOB_ID}_1 → Batch 1: panoramas 1278-2555"
echo "  • ${JOB_ID}_2 → Batch 2: panoramas 2556-3832"
echo "  • ${JOB_ID}_3 → Batch 3: panoramas 3833-5110"
echo ""
echo "================================================"
echo ""
echo "📊 Monitor jobs:"
echo "  squeue -u \$USER"
echo ""
echo "📝 Watch logs:"
echo "  tail -f logs/rectification_${JOB_ID}_0.out  # Batch 0"
echo "  tail -f logs/rectification_${JOB_ID}_1.out  # Batch 1"
echo "  tail -f logs/rectification_${JOB_ID}_2.out  # Batch 2"
echo "  tail -f logs/rectification_${JOB_ID}_3.out  # Batch 3"
echo ""
echo "📈 Check progress:"
echo "  watch -n 30 'find outputs/rectification_results -name \"*.jpg\" | wc -l'"
echo ""
echo "================================================"
echo "⏰ Estimated completion: ~17 hours (4× faster!)"
echo "================================================"
