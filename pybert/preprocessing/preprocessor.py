#encoding:utf-8
import re
import jieba

class Preprocessor(object):
    def __init__(self,min_len = 2,stopwords_path = None):
        self.min_len = min_len
        self.stopwords_path = stopwords_path
        self.reset()

    # jieba分词
    def jieba_cut(self,sentence):
        seg_list = jieba.cut(sentence,cut_all=False)
        return ' '.join(seg_list)

    # 加载停用词
    def reset(self):
        if self.stopwords_path:
            with open(self.stopwords_path,'r') as fr:
                self.stopwords = {}
                for line in fr:
                    word = line.strip(' ').strip('\n')
                    self.stopwords[word] = 1

    # 去除长度小于min_len的文本
    def clean_length(self,sentence):
        if len([x for x in sentence]) >= self.min_len:
            return sentence

    # 全角转化为半角
    def full2half(self,sentence):
        ret_str = ''
        for i in sentence:
            if ord(i) >= 33 + 65248 and ord(i) <= 126 + 65248:
                ret_str += chr(ord(i) - 65248)
            else:
                ret_str += i
        return ret_str

    #去除停用词
    def remove_stopword(self,sentence):
        words = sentence.split()
        x = [word for word in words if word not in self.stopwords]
        return " ".join(x)

    # 提取中文
    def get_china(self,sentence):
        zhmodel = re.compile("[\u4e00-\u9fa5]")
        words = [x for x in sentence if zhmodel.search(x)]
        return ''.join(words)
    # 移除数字
    def remove_numbers(self,sentence):
        words = sentence.split()
        x = [re.sub('\d+','',word) for word in words]
        return ' '.join([w for w in x if w !=''])

    def remove_whitespace(self,sentence):
        x = ''.join([x for x in sentence if x !=' ' or x !='' or x!='  '])
        return x
    # 主函数
    def __call__(self, sentence):
        x = sentence.strip('\n')
        x = self.full2half(x)
        # x = self.jieba_cut(x)
        # if self.stopwords_path:
        #     x = self.remove_stopword(x)
        x = self.remove_whitespace(x)
        x = self.get_china(x)
        x = self.clean_length(x)

        return x

if __name__ == "__main__":
    sentence = '沪基指全周大涨8.25% 创两个月最大周涨幅全景网2月6日讯 受到A股市场牛年高歌猛进影响，沪基指本周大涨逾8%，创两个月来最大周涨幅，强势迎来五连阳，两市封闭式基金近乎全线上扬，市场分析人士表示，封闭式基金有望继续上扬。两市基指本周日线呈现节节攀升走势，周一沪深基金指数小幅收高，迎来牛年开门红，沪基指收复2800点关口，深基指收复2900点关口，但仍跑输大盘，两市接近九成封闭式基金收盘上涨，因A股市场呈现普涨格局。周二沪深基金指数再度大幅收高，两市基指涨幅均逾1%，沪基指创出近四个月高点，接近九成封闭式基金上涨，因A股市场延续反弹。周三沪深基金指数连续第二日大涨，沪基指收复2900点关口，创2008年9月26日以来收盘高点，深基指收复3000点关口，双双跑赢大盘。两市封闭式基金近乎全线上扬，因政策利好预期刺激A股市场呈现普涨格局。周四沪深基金指数小幅收低，双双终结三连阳，盘中呈现冲高回落走势，两市超过八成封闭式基金反转收跌，因A股市场重拾跌势。周五沪深基金指数再度重拾升势，沪基指收复3000点关口，深基指亦收复3100点关口，两市封闭式基金近乎全线上扬，因外围市场提振A股强势反弹。截止收盘，沪深基指涨幅均超过7%，沪基指收复3000点关口，深基指则收复3100点关口。沪基指本周开盘于2801.81点，收盘于3018.63点，上涨230.12点或8.25%；深基指开盘于2894.36点，收盘于3101.81点，上涨221.39点或7.69%。两市基金全周的成交金额为110.75亿元，较上周放大超过两成，成交量为10332.3万手。股市方面，沪综指全周上涨9.57%，深成指上涨10.79%。单个基金方面，开盘交易的32只封闭式基金全线上涨，瑞福进取全周上涨13.93%，领涨封基，其他涨幅靠前的有建信优势上涨10.88%，基金裕泽上涨10.58%，基金通乾上涨10.55%，基金同益上涨10.19%，基金同盛上涨9.91%，基金金鑫上涨9.76%，基金鸿阳上涨9.39%，基金开元上涨9.08%，基金景福上涨8.60%，基金普丰上涨8.36%，基金泰和上涨8.23%；富国天丰成为唯一下跌的基金，跌幅为0.39%。国都证券基金研究员苏昌景认为，风险溢价较高的封基依然具有较好的长期投资价值。短线而言，未来短期交易价值将主要来源净值增长贡献，且折价率有增大的风险，对封基短期投资价值保持谨慎乐观，投资者可关注仓位较高和选股能力较强的创新型基金大成优选以及大盘基金基金安顺、基金科瑞、基金通乾、基金开元和基金同益。华泰证券亦认为，封闭基金尚有一波行情，但须密切关注内部收益率这个指标，该指标低于5%时应及时获利了解。同时应注意到，年末的封闭基金行情与分红预期存在很大的关系，最近上涨较快的基金均为具有分红潜力的品种，投资者应注意热点过后调整的风险。国海证券认为，谨慎积极的资产配置，结合市场热点的行业以及个股配置策略，封闭式基金2月份有望持续保持上涨态势。投资者2月份可适当关注“攻防有度”的封闭式基金。（全景网/雷鸣）'
    progress = Preprocessor()
    print(progress(sentence))