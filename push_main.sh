#!/bin/bash
# mainブランチにpushしてRunPodのEdit Endpointを自動で開く
git add -A
git commit -m "${1:-update}"
git push origin main
echo ""
echo "✅ Push完了"
echo "🔗 Edit Endpointを開いています..."
open "https://www.runpod.io/console/serverless/endpoint/ox5awa5oe3s2yw/edit"
