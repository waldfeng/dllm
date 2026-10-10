"""Create v1.4 while preserving the source DOCX package and formatting."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import math
from lxml import etree
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TITLE = '一种共享记忆的深度自适应大语言模型及其训练方法'
SOURCE = ROOT / '专利/一种共享记忆稀疏路由的自适应深度大语言模型及其训练方法_1.3_专利交底书.docx'
TARGET = ROOT / f'专利/{TITLE}_1.4_修正版.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
FONT = '/tmp/opencode/font/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
TEXT = {
0: TITLE,
8: '注：即申请技术的通用名称，不得使用商标、型号、人名、地名，不得超过25字。',
9: '名称' + TITLE,
11: '所属技术领域 计算机实现的大语言模型结构、共享参数访问及自适应计算深度训练。',
14: '背景技术 US2019/0266236A1公开NLP置信度早退；US2022/0157045A1公开中间状态门控及相邻状态比较；US2026/0148045A1公开生成模型间路由和冻结骨干后训练早退头。Mobius公开全局前馈记忆，MoEUT结合共享层与注意力、前馈专家，SwitchHead、CALM及LayerSkip亦提供相关常规基础。因此，拟保护内容收窄为第七部分的具体结构与训练组合。多渠道检索、权利要求核验及尚未取得全文的中国专利线索，另见《检索与代码核验记录_1.4》。',
17: '本发明解决的技术问题：在计算机中减少不同计算深度重复存储前馈及注意力投影参数，同时保留各深度独立选块的能力；使中间深度的输出能够参与深度评价，并让骨干适配各样本所选深度。为此，采用四类共享参数库、逐循环独立路由以及逐深度读出与骨干交替更新的组合。',
19: '1. 如图1，输入文本切分为词元（文本小单元），经嵌入转换为隐藏状态（内部向量）。模型依次执行L个循环；深度t指已执行t个循环。各循环有独立路由器及两个归一化层，共用前馈、查询Q、键值KV、输出O四类参数库，每库N块。这里“记忆”是存于计算机内存的可学习矩阵，不是历史文本。骨干包括嵌入、共享库、循环模块、最终归一化和共享语言模型头W；逐深度校准读出是骨干之外的输出修正模块。',
20: '2. 前馈块内的门、升维、降维三矩阵保持配对；KV块内K、V两矩阵保持配对。查询块和输出块各存一个矩阵。同一库的块供不同循环、不同词元访问，参数不随循环复制。L、N及每次选块数k分别设定，1≤k≤N。归一化、位置编码、因果注意力和残差相加采用常规方法。',
21: '3. 每循环先归一化状态，由本循环路由器为每个词元产生四组各N个得分。四分支分别选前馈、Q、KV、O库中得分最高的k块，用softmax将所选得分转为总和为1的权重。Q、K、V先分别投影并加权，再进行注意力计算；K、V使用同一组块号及权重，但使用不同矩阵。O分支处理注意力结果；前馈分支处理第一次残差相加并再次归一化后的状态。四分支的块号与权重均由第一次归一化状态确定，在该循环内沿用。',
22: '4. 如图1右侧，按块收集选中该块的词元，用共享矩阵计算，再按各词元的路由权重累加回原位置。前馈在每个选中块内先算SiLU(门投影)×升维投影，再降维，最后混合各块输出；×为逐元素相乘。这保持同块三矩阵的完整计算路径。逐块分派属于常规执行手段，在本方案中用于实现上述共享及配对结构，不单独作为创新。',
23: '5. 如图2(a)，同一训练流程对每个小批数据依次执行“全深度采集→逐样本选深度→A相更新读出→按原选深度分组→B相更新骨干”。A、B是每轮交替更新的两组参数，不是两次独立训练；当前实现要求梯度累积次数为1。骨干可随机初始化、由已有权重初始化或从联合检查点续训。',
24: '6. 采集时无梯度执行全部L个循环，缓存各深度最后一个输入位置的归一化状态h、状态变化δ及原头得分Wh。利用当前读出和真实下一词元，为每个样本分别计算各深度代价；在t_min至L中选代价最小深度，同值选较浅者。该离散深度不求梯度，并在本轮A、B更新中保持不变。A相使用缓存训练读出，不更新骨干；B相重新执行各样本所选深度内的骨干。',
25: '7. 深度t的读出得分为z_t=s_t(Wh_t+B_tA_th_t+c_t)。A_t将状态压缩到r维，B_t将压缩结果转为词表得分修正，c_t为词元偏置，s_t为得分缩放系数；得分经softmax转为概率q_t。B_t、c_t初始化为0，s_t初始化为1，使初始读出等于原头。各深度修正参数独立，W共享；同一读出用于深度评价和退出输出。A相只更新A_t、B_t、c_t、s_t；B相固定这些参数，但误差仍通过读出传回含W的骨干。',
26: '8. 六项代价均以σ(u)=1/(1+e^(−u))平滑映射到0至1：C₁=σ(−ln q_t(y)/ln V−1)，C₂=σ((τ_c−p_t)/τ_c)，C₃=σ((τ_m−m_t)/τ_m)，C₄=σ((δ_t−τ_d)/τ_d)，C₅=σ(t/L)，C₆=σ((t_min−t)/L)。y是真实下一词元，V是词表大小，p_t为最高预测概率，m_t为最高与次高概率之差；δ_t为相邻归一化状态差的长度除以当前状态长度，再对序列位置平均，分母至少10⁻⁶，采集首深度记0。τ_c、τ_m、τ_d均为正阈值。六项依次衡量预测误差、置信度不足、候选过近、状态不稳定、计算深度及过浅程度；数值越小越优。',
27: '9. 对样本i，代价ℓ_t,i=ΣⱼπⱼCⱼ,t,i，π为预设非负六项权重、和为1，训练中固定。仅在候选深度上取ρ_t,i=softmax_t(−ℓ_t,i/T)，T>0为固定选择温度。所选深度d_i=argmin_tℓ_t,i。A相损失为平均所选代价＋α·贡献对齐＋平均[−lnρ_d_i,i]＋β·选择熵。对齐项为预设π与softmax_j(ln(max(πⱼCⱼ,d_i,i,10⁻¹²)))的交叉熵；选择熵为−Σ_tρ_t,i lnρ_t,i/ln M，M为候选深度数，M=1时记0。α、β为非负系数。前两项优化所选深度读出，后两项用已选深度作固定目标，使候选深度分布集中；它们合并为一次A相更新。',
28: '10. A相更新后不重新选择深度。B相按原d_i分组，对各组重新执行前d_i个循环，使用该深度更新后的读出计算序列预测交叉熵，并加路由均衡损失，再更新骨干。读出参数固定，骨干中的共享库、已执行循环的路由器与归一化、嵌入及W可更新；未执行循环不参与该组计算。均衡项采用常规“全库平均路由概率×实际选块频率”方法，分别按四分支及已执行循环平均。',
29: '11. 如图2(b)，生成推理没有真实标签，舍去C₁，不对其余权重重新归一化。每循环以末位读出概率计算置信度和间隔，结合状态变化、深度及过浅项，将样本代价平均为整批停机代价。t<t_min时继续；从t_min起保存候选最小代价、深度和状态，严格下降则更新，首次不再下降（含相等）即停止，用保存深度的读出输出。持续下降则用第L深度输出。首循环没有相邻状态，推理略去状态变化项。实际执行数可能比返回深度多1；这是局部在线停机，不等于全深度最优选择。每个生成步骤重算当前序列，无KV缓存；关闭提前退出时跑满深度并使用原头。',
30: '12. 实施例：词表151665，隐藏维1536，前馈中间维4096，L=N=24，k=3，r=16；骨干约10.92亿参数，读出约6247万参数。可设t_min=8、τ_c=0.9、τ_m=0.3、τ_d=0.05、T=0.25，六项权重π=(0.65,0.05,0.05,0.05,0.15,0.05)，α=0.1、β=0.01，B相路由损失权重0.01；这些数值仅为实施参数。计算机保存联合参数及预设停机参数供推理加载。结构和流程以当前程序为依据，不将未经对照实测的准确率、速度或显存改善写作既成效果。',
33: '有益效果 四类参数库跨循环共享，使库参数量不随循环数重复增加，而各循环保留独立选块能力；KV及前馈块配对使被选矩阵按一致块身份参与计算。缓存全深度状态后单独更新读出，再按样本所选深度重算骨干，使骨干反向传播只经过该组执行的循环。在线停机可省去部分后续循环，但仍有读出及探测开销；训练仍执行全深度采集。k/N不代表显存缩减比例，退出深度比不代表实测速比。',
37: '关键点与欲保护点（仅列拟保护的区别组合，以可核实的现有技术和代码为依据）：',
38: '1. 模型结构主组合：前馈三矩阵、Q、配对KV及O组成四类跨循环共享参数库；各循环有独立四分支路由，四分支由注意力前同一状态产生并在循环内沿用；KV共用块号及权重，前馈每块完成完整路径后再混合。上述连接和选择关系作为整体限定，区别于只共享前馈库或仅做层内注意力专家的方案。',
39: '2. 结构组合的限定：Q、K、V分别先按选块加权形成投影结果，再进行注意力；O投影及前馈沿用本循环原选择，处理不同位置的状态。每循环独立路由与跨循环共享库共同限定，不把“四路”“KV绑定”或一般专家选择孤立主张。',
40: '3. 训练主组合：对上述模型无梯度缓存全深度末位状态及原头得分，先按样本确定深度；A相只更新逐深度读出，本轮选择保持不变；B相按原深度分组重算骨干，固定读出参数并将序列误差通过读出传回骨干。该执行顺序、缓存复用及参数更新关系共同限定，不以一般交替优化为保护内容。',
41: '4. 训练组合的读出限定：同一共享语言模型头叠加逐深度低秩修正、词元偏置和缩放，零修正初始化；该读出同时用于样本深度评价、B相序列监督及推理退出输出。低秩修正、零初始化或缩放本身属于已知手段，仅作为上述训练组合的限定。',
42: '5. 训练组合的深度监督限定：固定六项权重，候选深度下限约束；以更新前选定深度同时参与所选代价、贡献对齐及深度分布交叉熵，配合选择熵优化读出；随后沿用该深度训练骨干。重点是选定深度贯穿A、B更新的具体关联，不单独保护损失公式或softmax选深度。',
43: '6. 配套推理限定：在上述共享结构与同源读出上，舍去真实标签项，按整批五项代价维护保存状态；首次不再下降时停止并回到保存深度输出。训练按样本全轨迹选择，推理按整批局部停机，二者分别记载，不声称规则或分布完全一致。',
44: '7. 保护范围以实现技术效果所必需的连接、数据流和更新步骤限定；块数、循环数、秩及阈值作为实施例，不作为单独创新。逐块分派、均衡损失及检查点仅简述其配套作用。',
45: '8. 以上为交底书中的拟保护组合，不是已授权权利要求，也不因旧报告中“未见完整组合”而认定具有创造性。正式权利要求应基于同一总发明构思，纳入上述有贡献的结构及训练特征，并根据补充核实的专利全文确定范围。',
46: '常规技术简述：top-k、softmax、归一化、因果注意力、残差、交叉熵、负载均衡、低秩适配、参数交替更新及并行同步采用已知方法。撰写依据为《专利法》第2、22、26、31条、实施细则有关说明书及附图的规定，以及现行审查指南有关人工智能模型结构、训练步骤和技术贡献整体评价的要求。',
56: '图1  四类共享参数库、独立循环路由及同源读出结构',
58: '图2  逐样本选深度的A/B交替训练及整批在线停机流程',
}


class Diagram:
    def __init__(self, size):
        self.image = Image.new('RGB', (size[0]*2, size[1]*2), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def text(self, xy, text, size=25):
        self.draw.text(tuple(v*2 for v in xy), text,
                       font=ImageFont.truetype(FONT, size*2), fill='black')

    def box(self, rect, text, size=25):
        self.draw.rectangle(tuple(v*2 for v in rect), fill='white', outline='black', width=4)
        font = ImageFont.truetype(FONT, size*2)
        lines = text.split('\n')
        y = (rect[1]+rect[3]-len(lines)*(size+9))/2
        for line in lines:
            width = self.draw.textlength(line, font=font)/2
            assert width < rect[2]-rect[0]-16, (line, width, rect)
            self.draw.text(((rect[0]+rect[2]-width), y*2), line, font=font, fill='black')
            y += size+9

    def arrow(self, points):
        self.draw.line([(x*2,y*2) for x,y in points], fill='black', width=4)
        a,b = points[-2:]
        angle = math.atan2(b[1]-a[1], b[0]-a[0]); length = 13
        ends = [b, (b[0]-length*math.cos(angle-.5),b[1]-length*math.sin(angle-.5)),
                (b[0]-length*math.cos(angle+.5),b[1]-length*math.sin(angle+.5))]
        self.draw.polygon([(x*2,y*2) for x,y in ends], fill='black')

    def data(self):
        stream = BytesIO(); self.image.save(stream, format='PNG', dpi=(300,300))
        return stream.getvalue()


def figures():
    g = Diagram((1859,1259))
    g.text((40,15),'（a）模型：依次执行循环1…L；共享库相同，各循环路由器不同',29)
    g.box((40,65,500,155),'101 文本 → 词元嵌入 → 状态x',26)
    g.box((40,195,500,280),'102 本循环第一次归一化 n',26)
    g.arrow([(270,155),(270,195)])
    g.box((565,195,1090,315),'103 本循环独立路由器\n从n一次产生前馈 / Q / KV / O选择\n各分支：N得分 → top-k → 权重',25)
    g.arrow([(500,237),(565,237)])
    g.box((1160,65,1819,340),'104 全局共享参数库（各N块）\n前馈：门 / 升维 / 降维同块配对\nQ：查询矩阵；O：输出矩阵\nKV：键K、值V同块配对\n所有循环访问同一库；L与N可分别设置',26)
    g.arrow([(1090,237),(1160,237)])
    g.box((40,370,1090,505),'105 对n分别计算Q、K、V投影并各自加权求和\nKV共用块号及权重，K与V矩阵不同\nQ、K加入位置编码 → 因果注意力 → 结果a',27)
    g.arrow([(270,280),(270,370)]); g.arrow([(825,315),(825,370)])
    g.box((40,555,1090,670),'106 对a使用已选O块及权重投影、加权\n第一次残差：x₁ = x + O输出',27)
    g.arrow([(570,505),(570,555)])
    g.box((40,720,1090,875),'107 第二次归一化 n₂ = norm2(x₁)\n使用已选前馈块及权重（不重新路由）\n每块：SiLU(门投影) × 升维投影 → 降维\n各完整块输出加权混合',27)
    g.arrow([(570,670),(570,720)])
    g.box((40,925,1090,1020),'108 第二次残差：x₂ = x₁ + 前馈输出\nx₂ → 下一循环（下一循环有自己的路由器）',27)
    g.arrow([(570,875),(570,925)])
    g.box((1160,390,1819,635),'（b）所有库的逐块执行\n按块收集选中该块的词元\n→ 用该块共享矩阵计算\n→ 乘词元路由权重\n→ 累加回该词元原位置\n不物化每词元的专家矩阵副本',27)
    g.arrow([(1490,340),(1490,390)])
    g.box((1160,700,1819,1020),'109 每深度t的同源读出\nx₂ → 最终归一化h_t\n共享原头：Wh_t\n修正：B_tA_th_t + c_t\n相加再乘缩放s_t → 得分z_t\nsoftmax → 预测概率\n用于深度评价 / 所选深度输出',27)
    g.arrow([(1090,975),(1125,975),(1125,850),(1160,850)])
    g.box((40,1080,1819,1205),'图例：实线箭头表示数据或选块控制流；103→104表示按块寻址。\n四类库跨循环共享；路由器、两次归一化及读出修正按深度独立。\nB_t、c_t=0，s_t=1初始化：初始读出等于原头；低秩A_t维度为r。',26)

    f = Diagram((1859,1500))
    f.text((40,12),'（a）联合训练：每小批一次A相更新、一次B相更新；按样本选深度',29)
    f.box((40,65,550,175),'201 初始化 / 续训\n骨干 + 逐深度读出',27)
    f.box((655,65,1819,175),'202 小批输入 → 无梯度运行全部L循环\n缓存末位归一化状态、状态变化、原头得分Wh',27)
    f.arrow([(550,120),(655,120)])
    f.box((655,220,1819,360),'203 当前读出 → 各样本各深度六项代价\n六项权重固定；候选仅t_min…L\n逐样本选最小代价深度d_i；同值取较浅；选择不求梯度',27)
    f.arrow([(1230,175),(1230,220)])
    f.box((655,405,1819,580),'204 A相：固定骨干，只更新逐深度读出\n复用缓存；所选代价 + 贡献对齐\n+ 所选深度作目标的深度分布交叉熵 + 选择熵\n合并一次更新；固定六项权重、选择温度及本轮d_i',27)
    f.arrow([(1230,360),(1230,405)])
    f.box((655,625,1819,810),'205 B相：按原d_i分组，重算各组前d_i循环\n使用A相更新后的读出，但固定其参数\n序列交叉熵 + 路由均衡 → 更新骨干（含共享库、W等）\n误差穿过固定读出；d_i不重选；本轮结束后进入下一小批',27)
    f.arrow([(1230,580),(1230,625)])
    f.box((40,220,550,390),'参数分组\nA：A_t、B_t、c_t、s_t\nB：读出之外全部骨干\n六项权重π：预设、固定',27)
    f.box((40,450,550,590),'本轮深度保持不变\nA更新前选择d_i\nA更新后继续沿用d_i',27)
    f.box((40,650,550,810),'206 保存联合参数\n保存π、T及模型配置\n下一小批重复采集和更新',27)
    f.arrow([(655,720),(550,720)])
    f.arrow([(290,810),(290,845),(605,845),(605,55),(1230,55),(1230,65)])
    f.text((40,880),'训练：全深度采集 + 分组截断反向传播。训练选择有标签，不等于推理停机。',27)
    f.draw.line((80,1870,3638,1870),fill='black',width=3)
    f.text((40,950),'（b）生成推理：整批共用退出深度；在当前输入序列上逐循环探测',29)
    f.box((40,1005,550,1150),'207 执行第t循环\n末位读出概率 + 状态变化\n无标签；无KV缓存',27)
    f.box((655,1005,1195,1150),'208 t < t_min？\n是：继续下一循环\n否：计算五项整批平均代价',26)
    f.arrow([(550,1075),(655,1075)])
    f.box((1290,1005,1819,1190),'209 首次候选或代价严格下降？\n是：保存代价、深度、状态\n未满L：继续；满L：输出\n否（含相等）：停止探测',25)
    f.arrow([(1195,1075),(1290,1075)])
    f.arrow([(925,1150),(925,1195),(150,1195),(150,1150)])
    f.arrow([(1290,1155),(1240,1155),(1240,1220),(290,1220),(290,1150)])
    f.box((655,1260,1819,1385),'210 用保存深度的同一读出输出下一词元\n停止时可比返回深度多算1循环；持续下降则返回L\n舍去标签项且不重归一化权重；首循环略去状态变化项',27)
    f.arrow([(1560,1190),(1560,1260)])
    f.box((40,1260,550,1385),'关闭提前退出\n完整执行L循环\n使用原头输出',27)
    f.text((40,1430),'生成下一词元后，把它加入输入序列再执行上述推理；局部停机不保证全深度最优。',27)
    return {'word/media/image1.png':g.data(), 'word/media/image2.png':f.data()}


def main():
    assert len(TITLE) == 23, len(TITLE)
    media = figures()
    with ZipFile(SOURCE) as src:
        original = etree.fromstring(src.read('word/document.xml'))
        root = etree.fromstring(src.read('word/document.xml'))
        paragraphs = root.find('w:body',NS).findall('w:p',NS)
        for index,text in TEXT.items():
            nodes = paragraphs[index].findall('.//w:t',NS)
            lengths = [len(n.text or '') for n in nodes]; total = sum(lengths)
            assert nodes
            offset = cumulative = 0
            for i,(node,length) in enumerate(zip(nodes,lengths)):
                cumulative += length
                end = len(text) if i == len(nodes)-1 else round(len(text)*cumulative/max(1,total))
                node.text = text[offset:end]; offset = end
                node.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
        xml = etree.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
        with ZipFile(TARGET,'w') as dst:
            for entry in src.infolist():
                dst.writestr(entry, xml if entry.filename == 'word/document.xml' else media.get(entry.filename,src.read(entry.filename)))
        for tree in (original,root):
            for node in tree.findall('.//w:t',NS):
                node.text = ''; node.attrib.pop('{http://www.w3.org/XML/1998/namespace}space',None)
        assert etree.tostring(original) == etree.tostring(root), 'format structure changed'
    with ZipFile(SOURCE) as src, ZipFile(TARGET) as dst:
        assert dst.testzip() is None
        assert src.namelist() == dst.namelist()
        for name in src.namelist():
            if name not in media and name != 'word/document.xml':
                assert src.read(name) == dst.read(name), name
    for i,content in enumerate(media.values(),1):
        (ROOT / f'专利/附图{i}_1.4.png').write_bytes(content)
    from retrieve_patents_v14 import IDS
    with ZipFile(ROOT / '专利/检索证据_1.4.zip','w') as evidence:
        for number in IDS:
            for suffix in ('.pdf','_ocr.txt'):
                path = Path('/tmp/opencode') / f'US{number}A1{suffix}'
                assert path.exists(), path
                evidence.write(path,path.name)
        evidence.write(ROOT / '专利/检索与代码核验记录_1.4.md','检索与代码核验记录_1.4.md')
    print(TARGET)
    print('Verified: original styles, sections, tables, drawing extents and package structure preserved.')


if __name__ == '__main__':
    main()
