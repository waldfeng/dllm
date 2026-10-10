"""Revise text and figures while retaining the 1.2 Word formatting."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
from lxml import etree
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TITLE = '一种共享记忆稀疏路由的自适应深度大语言模型及其训练方法'
SOURCE = ROOT / '专利/一种共享记忆稀疏路由大模型两阶段训练及提前退出方法_1.2_核实修订.docx'
TARGET = ROOT / f'专利/{TITLE}_1.3_专利交底书.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
TEXT = {
0: TITLE,
9: '名称' + TITLE,
11: '所属技术领域 大语言模型、共享参数、稀疏路由及自适应计算深度训练。',
14: '背景技术 参数共享、混合专家和提前退出已有公开研究，如Universal Transformer、SwitchHead、CALM及LayerSkip。本方案不单独保护这些一般方法，重点是跨循环统一共享记忆、独立四路路由与两相联合训练的组合。',
17: '本发明解决的技术问题：使不同计算深度共用前馈和注意力投影参数，又能独立选择参数块；同时使中间深度具备预测能力，按输入选择计算深度。模型采用统一共享记忆与逐循环独立路由，在同一训练流程内交替更新校准读出和骨干，不将两相分成先后独立的训练阶段。',
19: '1. 如图1，模型由词元嵌入、全局共享记忆、L个循环推理器、最终归一化、共享语言模型头和逐深度校准读出组成。词元是文本切分单元，隐藏状态是模型内部的向量表示，深度是已执行的循环数，骨干指读出以外的基础模型。记忆是可学习参数库，不是历史文本库；分为前馈、查询、键值和输出四库，每库N块。各循环有独立路由器和两个归一化层，共用同一记忆。L为最大循环数，N为每库块数，k为每次选块数，可分别设置；实施例L=N=24、k=3。',
20: '2. 每个前馈块包含配对的门、升维和降维矩阵；查询块和输出块各含一个投影矩阵；键值块含配对的键、值矩阵。矩阵把状态转换成不同计算表示，同一块可被不同循环和词元使用。归一化、位置编码、因果注意力和残差相加采用常规方法。',
21: '3. 每循环先归一化状态，再由自身路由器产生4×N个得分。四个分支分别对前馈、查询、键值和输出库选得分最高的k块；softmax将所选得分转为总和为1的权重，计算结果按权重求和，不是简单平均。键和值使用相同块号和权重、不同矩阵。路由选择在该循环内复用：查询、键、值处理注意力前状态；输出投影处理注意力结果；前馈处理残差相加后再次归一化的状态。',
22: '4. 按块收集选中该块的词元，计算后按权重累加回原位置，不为每个词元复制所选矩阵。前馈在同块内完成SiLU(门投影)与升维投影逐元素相乘、再降维，最后混合各块输出；门投影控制特征通过，升维与降维扩大和恢复向量维度。',
23: '5. 单阶段两相联合训练如图2：同一流程反复执行“采集状态→A相更新读出→选择深度→B相更新骨干”。A、B是每轮的两个更新步骤，不是两次独立训练。当前程序从已有骨干检查点初始化，或从联合检查点续训；本实施例不表示已实现随机初始化直接训练。',
24: '6. 每份小批数据先无梯度运行全部L个循环，记录各深度最后一个输入词元的归一化状态h_t、相邻状态变化及原头得分，骨干参数不更新。A相用真实下一词元训练读出与六项组合权重；更新后重算代价，选择深度。B相重新运行所选深度内的骨干，固定读出参数、更新骨干。两相分别更新各自参数，骨干不是全程冻结。',
25: '7. 深度t的读出为z_t=s_t(W h_t+B_t A_t h_t+c_t)。h_t为归一化状态，z_t为各词元的得分；W为共享语言模型头；A_t把状态压缩到r维，B_t把压缩结果转换为得分修正；c_t为词元偏置，s_t为得分缩放系数，实施例r=16。B_t=c_t=0、s_t=1初始化，使初始读出等于原头。得分经softmax转为概率，用于深度评价及退出输出。A相只更新A_t、B_t、c_t、s_t和组合权重；B相固定这些参数，更新含W的骨干，误差仍经读出传回骨干。',
26: '8. A相六项代价越小越优：①C₁,t=−ln q_t(y)/ln d，降低预测误差；②C₂,t=[τ_c−p_t]₊，惩罚置信度不足；③C₃,t=[τ_m−m_t]₊，惩罚候选概率过近；④C₄,t=[δ_t−τ_d]₊，惩罚状态变化过大；⑤C₅,t=t/L，倾向少算循环；⑥C₆,t=[t_min−t]₊/L，惩罚过浅深度。[u]₊=max(0,u)，ln为自然对数；q_t(y)为全词表中真实下一词元y的概率，d为隐藏维度；p_t为最高概率，m_t为最高与次高概率之差；δ_t为相邻归一化状态差的向量长度除以当前状态长度（分母至少10⁻⁶），在有效位置取平均，首深度记0；τ_c、τ_m、τ_d为置信度下限、间隔下限和状态变化上限；t_min为最小退出深度。',
27: '9. 批内平均代价为ℓ_t=ΣⱼπⱼC̄ⱼ,t，j=1至6，横线表示样本平均；π=softmax(u)，u为可训练六维参数，πⱼ非负且和为1。选择分布ρ=softmax(−ℓ/T)，T>0为固定选择温度，区别于读出缩放s_t。A相损失J_A=Σⱼaⱼ+J_align+βH，aⱼ=πⱼ⟨Cⱼ⟩，尖括号表示样本及深度平均；J_align=sigmoid(CE(ln(max(a,10⁻¹²)),η))，CE为交叉熵、sigmoid把结果映射到0至1、η为固定六项先验比例，用于引导各项贡献；H=−Σ_tρ_t lnρ_t/ln L，β为其权重，降低H使选择集中。此处ρ覆盖全部深度。A相固定骨干，前三项直接影响读出，后三项参与评价和权重学习。',
28: '10. A相更新后，用新读出和新权重重算ℓ_t，在t_min至L中取最小代价深度t*；等价于在这些候选上取softmax(−ℓ/T)的最大值，相同代价取较浅者。t*是本轮选定深度，不对离散选择本身求梯度。B相重新执行前t*个循环，以J_B=CE_seq(t*)+λR(t*)更新骨干：CE_seq为所选深度校准读出的序列交叉熵，衡量预测误差；R为前t*循环的平均路由均衡损失，减少块使用过于集中；λ为权重，实施例0.01。每个路由分支的均衡项为NΣ_bI_bF_b，b为块号，I_b为平均全库路由概率，F_b为按词元及选块槽位统计的分配频率，再对分支及循环平均。',
29: '11. 推理不使用真实标签项，只计算ℓ_t^推理=Σⱼ₌₂⁶πⱼC̄ⱼ,t。整批共同选深度：从t_min起保存最小代价和状态，代价严格下降则更新，首次不再下降（含相等）即停止，用保存深度的读出输出；持续下降则运行全部循环。实际循环数可比返回深度多1。这是在线局部停机，与训练全轨迹选深度不同，不保证未计算深度没有更优结果。每次生成均在当前输入序列上重算，当前实现无键值缓存；关闭退出则运行满深度并用原头输出。',
30: '12. 实施与验证：骨干约10.92亿参数，隐藏维1536，前馈中间维4096，24循环、每库24块、每头选3块；秩16读出约6247万参数。程序已实现两组优化器、更新后重选深度、所选深度的骨干反向传播及联合检查点保存。多卡时各卡可选不同深度，同步梯度前将未产生的骨干梯度补零。为清楚体现每轮A/B更新，实施可采用梯度累积次数1；不将现有多微批实现表述为严格等效累积。冻结读出的旧评估不作为本联合方法效果；质量和速度须对联合检查点实测。',
33: '有益效果 共享记忆使前馈和注意力投影不随循环复制，独立路由保留各深度的选择能力；块配对避免键值或前馈矩阵不匹配。两相交替使读出学习各深度预测，骨干适配所选深度，固定另一组参数使更新职责清楚。推理可跳过部分后续循环，但训练仍有全深度状态采集；k/N仅表示单词元每分支选块比例，不代表显存同比减少，深度比不等于实测加速。',
37: '关键点与欲保护点（以具体组合为重点，常规算法和数值不单独主张）：',
38: '1. 全局统一参数库与独立循环路由：前馈三矩阵、查询、键值及输出投影按块组织并跨循环共享，各循环使用自身路由器，库容量与循环数可分别设置。',
39: '2. 四域路由与配对执行：四分支分别选择前馈、查询、键值和输出块，键值绑定块号及权重，前馈在同块内完成门控、升维和降维后再混合；与统一记忆共同保护。',
40: '3. 逐块执行：收集选块词元、用共享块计算、按权重累加，避免为词元复制专家矩阵；作为上述路由的实现限定。',
41: '4. 逐深度同源读出：共享头叠加低秩修正、词元偏置和缩放，恒等初始化，同一读出用于深度评价和退出输出。',
42: '5. 单阶段两相更新：每轮完整无梯度采集状态，A相仅更新读出及权重，更新后重选深度，B相固定读出、仅更新所选深度内的骨干；误差通过固定读出传回骨干。',
43: '6. 配套代价与深度选择：六项代价、贡献先验和选择熵用于A相；以新参数在最小深度以上选代价最小者，再以序列预测误差与均衡项适配骨干。',
44: '7. 在线停机：推理舍去标签项，在最小深度以上记录最小代价及状态，首次不再下降时返回保存深度，持续下降则跑满；与同源读出结合保护。',
45: '8. 保护重点是“统一记忆＋独立四路路由”与“同源读出＋更新后选深度＋两组参数交替更新”的配合。训练全轨迹选择和推理局部停机分别限定，不以一般参数共享、低秩修正或交替优化单独主张创新。',
46: '常规技术简述：top-k、softmax、归一化、交叉熵、负载均衡、低秩适配、并行同步及检查点沿用已知方法。公开背景可参考SwitchHead（arXiv:2312.07987）、CALM（arXiv:2207.07061）、LayerSkip（arXiv:2404.16710）；具体专利差异以可核实公开文本为准。',
56: '图1  共享记忆四路路由与逐深度读出结构',
58: '图2  单阶段两相联合训练及在线推理流程',
}

FONT = '/tmp/opencode/font/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
class Diagram:
    def __init__(self, size):
        self.image = Image.new('RGB', size, 'white')
        self.draw = ImageDraw.Draw(self.image)
    def text(self, xy, text, size=24):
        self.draw.text(xy,text,font=ImageFont.truetype(FONT,size),fill='#172333')
    def box(self, rect, text, fill='#f3f5f8', size=24):
        self.draw.rounded_rectangle(rect,12,fill=fill,outline='#263548',width=3)
        font=ImageFont.truetype(FONT,size); lines=text.split('\n'); y=(rect[1]+rect[3]-len(lines)*(size+9))//2
        for line in lines:
            length=self.draw.textlength(line,font=font)
            assert length < rect[2]-rect[0]-14, (line, length)
            self.draw.text(((rect[0]+rect[2]-length)/2,y),line,font=font,fill='#172333'); y+=size+9
    def arrow(self, points):
        import math
        self.draw.line(points,fill='#263548',width=3)
        a,b=points[-2:]; angle=math.atan2(b[1]-a[1],b[0]-a[0]); l=14
        self.draw.polygon([b,(b[0]-l*math.cos(angle-.5),b[1]-l*math.sin(angle-.5)),(b[0]-l*math.cos(angle+.5),b[1]-l*math.sin(angle+.5))],fill='#263548')
    def data(self):
        out=BytesIO(); self.image.save(out,format='PNG'); return out.getvalue()

def figures():
    g=Diagram((1859,1259))
    g.text((45,20),'共享记忆模型：展开显示一个循环（各循环独立路由，共用同一参数库）',29)
    g.box((45,85,425,175),'词元 → 嵌入 → 当前状态 x')
    g.box((45,220,425,310),'第一次归一化：n = norm1(x)')
    g.box((510,205,990,330),'本循环独立路由器\n前馈 / 查询 / 键值 / 输出\n各选 Top-k + softmax','#fff4c4')
    g.arrow([(235,175),(235,220)]);g.arrow([(425,265),(510,265)])
    g.box((1125,85,1810,390),'全局共享记忆（各库 N 块）\n前馈：同块门 / 升维 / 降维矩阵\n查询：Q 矩阵\n键值：K、V 矩阵（不同矩阵）\n输出：O 矩阵\n各循环访问同一库；块参数不按循环复制','#deecfb')
    for rect,text in [((510,380,730,475),'查询分支\nQ 块号 / 权重'),((765,380,990,475),'键值分支\nK、V共用路由')]:g.box(rect,text,'#deecfb',22)
    g.arrow([(620,330),(620,380)]);g.arrow([(875,330),(875,380)])
    g.box((45,525,990,645),'Q、K、V分别对 n 按所选块投影并加权求和\nQ、K加入位置编码 → 因果注意力 → 注意力结果 a','#deecfb')
    g.arrow([(235,310),(235,525)]);g.arrow([(620,475),(620,525)]);g.arrow([(875,475),(875,525)])
    g.box((45,695,990,790),'沿用路由器输出分支已选的 O 块与权重，对 a 投影并加权求和\n第一次残差：x₁ = x + 注意力输出','#deecfb')
    g.arrow([(520,645),(520,695)])
    g.box((45,840,990,960),'第二次归一化：n₂ = norm2(x₁)\n前馈分支沿用本循环已选块和权重\n每块：SiLU(门投影) × 升维投影 → 降维；再加权求和','#def5e6')
    g.arrow([(520,790),(520,840)])
    g.box((45,1010,990,1110),'第二次残差：x₂ = x₁ + 前馈输出\nx₂传给下一循环；不共享各循环的路由器')
    g.arrow([(520,960),(520,1010)])
    g.box((1125,455,1810,630),'稀疏计算（所有投影及前馈）\n按块收集选中的词元 → 用该块计算\n按路由权重累加回原位置\n不复制“词元 × k × 输入维 × 输出维”矩阵')
    g.arrow([(1460,390),(1460,455)])
    g.box((1125,695,1810,950),'每个深度 t 的可选输出路径\n对 x₂作最终归一化得到 h_t\n原头 W h_t + 低秩修正 B_t A_t h_t + c_t\n乘以 s_t → 得分 z_t → softmax概率\n同一读出用于深度评价和所选深度输出','#ffe9c7')
    g.arrow([(990,1060),(1050,1060),(1050,820),(1125,820)])
    g.text((60,1175),'顺序：注意力 → 残差 → 再归一化 → 前馈 → 残差；各分支选择均来自第一次归一化状态。',24)
    g.text((1140,1020),'L=24，N=24，k=3 为实施例参数。',24)
    g.text((1140,1070),'记忆块数 N 与循环数 L 可分别设置。',24)
    g.text((1140,1120),'各深度读出参数独立，原头 W 共用。',24)

    f=Diagram((1859,1500))
    f.text((45,20),'（a）同一训练流程内反复进行 A / B 两相更新',29)
    f.box((45,85,535,205),'初始化 / 续训检查点\n骨干 + 逐深度读出 + 六项权重',size=25)
    f.box((660,85,1810,205),'小批数据 → 全部 L 循环无梯度前向\n记录末位状态、相邻变化、原头得分；不更新骨干',size=25)
    f.arrow([(535,145),(660,145)])
    f.box((660,260,1810,415),'A 相：只更新读出和六项权重，骨干固定\n六项：预测误差 / 置信度不足 / 间隔不足 / 状态变化 / 深度 / 过浅守卫\n损失：六项加权均值 + 先验对齐 + 选择熵（覆盖全部深度）\n使用末位真实下一词元；固定先验和选择温度','#fff4c4',25)
    f.arrow([(1235,205),(1235,260)])
    f.box((660,470,1810,595),'更新后重新计算各深度读出和六项代价\n只在 t_min 至 L 中选最小代价深度 t*（同值取较浅者）\n离散选深度本身不求梯度','#ffe9c7',25)
    f.arrow([(1235,415),(1235,470)])
    f.box((660,650,1810,835),'B 相：重新执行前 t* 循环，读出参数固定\n所选深度读出 → 序列交叉熵 + 前 t* 循环平均路由均衡损失\n误差经过固定读出传回骨干 → 更新记忆、路由器、归一化、嵌入及 W\n读出和六项权重不在 B 相更新','#def5e6',25)
    f.arrow([(1235,595),(1235,650)])
    f.box((45,470,535,655),'更新对象分组\nA：读出 + 组合权重\nB：全部骨干（含 W）\n两组优化器，各自更新',size=25)
    f.box((45,710,535,835),'下一小批重复上述流程\n保存联合参数及优化器状态',size=24)
    f.arrow([(660,765),(535,765)]);f.arrow([(535,735),(600,735),(600,50),(1235,50),(1235,85)])
    f.text((45,870),'单阶段指一个交替训练流程；当前代码从已有检查点开始。实施可采用累积次数1。',24)
    f.draw.line([(45,925),(1810,925)],fill='#99a4b2',width=2)
    f.text((45,955),'（b）生成推理：无真实标签，逐循环在线停机（与训练全轨迹选深度不同）',28)
    f.box((45,1020,535,1150),'当前输入序列 → 执行循环 t\n计算状态变化与读出概率\n每次生成重新计算，无 KV 缓存',size=23)
    f.box((660,1020,1190,1150),'t < t_min：继续循环\nt ≥ t_min：计算五项停机代价\n舍去训练中的真实标签误差项',size=23)
    f.arrow([(535,1085),(660,1085)])
    f.box((1315,1020,1810,1190),'首次候选或代价严格下降？\n是：保存最优代价、深度和状态\n未满 L 则继续；满 L 则输出\n否（含相等）：立即停止',size=23)
    f.arrow([(1190,1085),(1315,1085)])
    f.box((660,1260,1810,1385),'用保存深度的同一校准读出生成下一词元\n首次不再下降时，可能已多算一个探测循环；持续下降则满深度\n关闭提前退出：运行全部循环并使用原始头（不使用校准读出）','#ffe9c7',24)
    f.arrow([(1570,1190),(1570,1260)])
    f.arrow([(1315,1160),(1250,1160),(1250,1215),(290,1215),(290,1150)])
    f.text((45,1430),'图中分组说明正常更新流程；异常跳步、通信和检查点属于常规工程处理。',23)
    return {'word/media/image1.png':g.data(),'word/media/image2.png':f.data()}

def main():
    media=figures()
    with ZipFile(SOURCE) as src:
        original=etree.fromstring(src.read('word/document.xml'))
        root=etree.fromstring(src.read('word/document.xml'))
        paragraphs=root.find('w:body',NS).findall('w:p',NS)
        for index,text in TEXT.items():
            nodes=paragraphs[index].findall('.//w:t',NS)
            lengths=[len(n.text or '') for n in nodes]; total=sum(lengths)
            assert nodes
            offset=cumulative=0
            for i,(node,length) in enumerate(zip(nodes,lengths)):
                cumulative+=length; end=len(text) if i==len(nodes)-1 else round(len(text)*cumulative/max(1,total))
                node.text=text[offset:end];offset=end
                node.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
        document=etree.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
        with ZipFile(TARGET,'w') as dst:
            for entry in src.infolist():
                dst.writestr(entry,document if entry.filename=='word/document.xml' else media.get(entry.filename,src.read(entry.filename)))
        for tree in (original,root):
            for node in tree.findall('.//w:t',NS):
                node.text='';node.attrib.pop('{http://www.w3.org/XML/1998/namespace}space',None)
        assert etree.tostring(original)==etree.tostring(root),'format structure changed'
    with ZipFile(SOURCE) as src,ZipFile(TARGET) as dst:
        assert dst.testzip() is None
        assert src.namelist()==dst.namelist()
        for name in src.namelist():
            if name in media:
                assert Image.open(BytesIO(src.read(name))).size==Image.open(BytesIO(dst.read(name))).size
            elif name!='word/document.xml':assert src.read(name)==dst.read(name),name
    print(TARGET)
    print('Verified: original formatting, image dimensions and package structure preserved.')

if __name__=='__main__':main()
