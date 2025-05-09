\rg --no-ignore \
    --regexp "(^import .*)|(from .*import.*)" \
    --no-line-number \
    --no-column \
    --glob src/**/*.py \
    --glob scripts/**/*.py \
    --glob tests/**/*.py \
    . \
    | cut -d ':' -f 2 \
    | cut -d '#' -f 1 \
    |  sort | uniq \
    | cut -d ' ' -f 2 \
    | sort | uniq | grep --invert src \
    > requirements.poetry.txt
    
