#!/usr/bin/env bash

# skip based on checksum
# rsync -r --checksum --info=progress2 mirror.ismll.de:~/KIWI/neurips-2022/experiments/configs/ configs/
# skip based on timestamp
rsync -r --update --info=progress2 mirror.ismll.de:~/KIWI/tsdm/experiments/kiwi/configs/     configs/
rsync -r --update --info=progress2 mirror.ismll.de:~/KIWI/tsdm/experiments/kiwi/slurm/       slurm/
rsync -r --update --info=progress2 mirror.ismll.de:~/KIWI/tsdm/experiments/kiwi/logs/        logs/
rsync -r --update --info=progress2 mirror.ismll.de:~/KIWI/tsdm/experiments/kiwi/results/     results/
rsync -r --update --info=progress2 mirror.ismll.de:~/KIWI/tsdm/experiments/kiwi/tensorboard/ tensorboard/
rsync -r --update --info=progress2 mirror.ismll.de:~/KIWI/tsdm/experiments/kiwi/checkpoints/ checkpoints/
