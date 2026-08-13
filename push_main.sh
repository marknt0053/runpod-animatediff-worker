#!/bin/bash
# mainブランチにpushしてRunPodのServerlessページを自動で開く
git add -A
git commit -m "${1:-update}"
git push origin main
echo ""
echo "✅ Push完了"
echo "🔗 RunPodのServerlessページを開いています..."
echo "📌 ox5awa5oe3s2yw (runpod-animatediff-strong-worker) のManage→Edit Endpointを操作してください"
open "https://www.runpod.io/console/serverless"
