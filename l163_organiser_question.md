主旨：Problem C（cadc1075）— Final 評估環境的相依套件

您好，

準備 Final 提交時，發現兩份官方文件對相依套件的說明不一致，想請確認三點。

**1. Final 評估環境是否預裝 scipy？**

Beta 投稿指南第 2 節：
> "Leave requirements.txt EMPTY (zero bytes). The contest evaluation
> environment already provides: numpy, torch, scipy, numba, tqdm, shapely,
> threadpoolctl..."

Beta 評估報告第 2(a) 節：
> "Several submissions import packages such as torch-geometric, torch-scatter,
> and scipy... Do not assume any package beyond the Python standard library
> is available."

我們的 optimizer 使用 scipy。請問 Final 環境是否預裝 scipy？
若能一併提供預裝套件與版本清單（特別是 scipy 與 numpy），我們就不必臆測。

**2. 無網路環境下，非空的 requirements.txt 如何安裝？**

系統規格頁載明 `Internet access: No`。若我們提供完整的 requirements.txt，
評估環境是否會透過本地鏡像執行 pip install？或者反而會導致安裝失敗？

**3. 提交檔中夾帶第三方套件是否被允許？**

作為備案，我們考慮在提交檔中夾帶所需套件（約 120 MB）。這是否符合規範？
投稿指南提到「禁止未使用的大型 binary（違反可能導致 DQ）」——若該套件在
環境已提供該套件時不會被載入，是否會被視為「未使用」？

感謝協助。

cadc1075
