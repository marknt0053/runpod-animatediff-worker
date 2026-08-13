#!/bin/bash
# strongブランチにpushしてRunPodのEdit Endpointを自動で開く
git add -A
git commit -m "${1:-update}"
git push origin strong
echo ""
echo "✅ Push完了"
echo "🔗 Edit Endpointを開いています..."
open "https://www.runpod.io/console/serverless/endpoint/cz74u1nurpeke6/edit"
