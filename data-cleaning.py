from math import sqrt
import numpy as np
import pandas as pd

# so tables are not truncated
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

path_to_repo = "C:\\Users\\abhil\\Documents\\Election Stats\\Political Spectrum\\refactor\\"

# delete
data = pd.read_csv(path_to_repo +
                   "anes_timeseries_2020_csv_20220210.csv")

data = data.drop(data.columns[[15, 17, 18, 19, 21, 22, 23, 25,
                 26, 27, 29, 30, 31, 33, 34, 35, 37, 38, 1508, 1509]], axis=1)
# get rid of some columns with bad data types

weights = pd.to_numeric(
    data['V200010b'], errors='coerce').replace(float('nan'), 0.0)

ideo_questions = {
    'V201234': ('PRE: GOVERNMENT RUN BY A FEW BIG INTERESTS OR FOR BENEFIT OF ALL', 'Would you say the government is pretty much run by a few big interests looking out for themselves or that it is run for the benefit of all the people?', "govt run for benefit of all people,"),
    'V201235': ('PRE: DOES GOVERNMENT WASTE MUCH TAX MONEY', 'Do you think that people in government waste a lot of the money we pay in taxes, waste some of it, or don’t waste very much of it?', "govt doesn't waste much tax money,"),
    'V201236': ('PRE: HOW MANY IN GOVERNMENT ARE CORRUPT', 'How many of the people running the government are corrupt?', "none in govt corrupt,"),
    'V201237': ('PRE: HOW OFTEN CAN PEOPLE BE TRUSTED', 'Generally speaking, how often can you trust other people?', "people never can be trusted,"),
    'V201238': ('PRE: ELECTIONS MAKE GOVERNMENT PAY ATTENTION', 'How much do you feel that having elections makes the government pay attention to what the people think?', "elections don't make govt pay attention,"),
    'V201246': ('PRE: 7PT SCALE SPENDING & SERVICES: SELFPLACEMENT', 'Some people think the government should provide fewer services even in areas such as health and education in order to reduce spending. Suppose these people are at one end of a scale, at point 1. Other people feel it is important for the government to provide many more services even if it means an increase in spending. Suppose these people are at the other end, at point 7. Where would you place yourself on this scale, or haven’t you thought much about this?', "govt provide many more services,"),
    'V201249': ('PRE: 7PT SCALE DEFENSE SPENDING: SELF-PLACEMENT', 'Some people believe that we should spend much less money for defense. Suppose these people are at one end of a scale, at point 1. Others feel that defense spending should be greatly increased. Suppose these people are at the other end, at point 7. Where would you place yourself on this scale, or haven’t you thought much about this?', "more defense spending,"),
    'V201252': ('PRE: 7PT SCALE GOV-PRIVATE MEDICAL INSURANCE SCALE: SELF-PLACEMENT', 'There is much concern about the rapid rise in medical and hospital costs. Some people feel there should be a government insurance plan which would cover all medical and hospital expenses for everyone. Suppose these people are at one end of a scale, at point 1. Others feel that all medical expenses should be paid by individuals through private insurance plans like Blue Cross or other company paid plans. Suppose these people are at the other end, at point 7.', "private healthcare,"),
    'V201255': ('PRE: 7PT SCALE GUARANTEED JOB-INCOME SCALE: SELF-PLACEMENT', 'Some people feel the government in Washington should see to it that every person has a job and a good standard of living. Suppose these people are at one end of a scale, at point 1. Others think the government should just let each person get ahead on their own. Suppose these people are at the other end, at point 7.', "govt shouldn't provide guaranteed jobs & ubi,"),
    'V201258': ('PRE: 7PT SCALE GOV ASSISTANCE TO BLACKS SCALE: SELF-PLACEMENT', 'Some people feel that the government in Washington should make every effort to improve the social and economic position of blacks. Suppose these people are at one end of a scale, at point 1. Others feel that the government should not make any special effort to help blacks because they should help themselves. Suppose these people are at the other end, at point 7.', "govt shouldnt help black people,"),
    'V201262': ('PRE: 7PT SCALE ENVIRONMENT-BUSINESS TRADEOFF: SELF-PLACEMENT', 'Some people think we need much tougher government regulations on business in order to protect the environment. Suppose these people are at one end of a scale, at point 1. Others think that current regulations to protect the environment are already too much of a burden on business. Suppose these people are at the other end, at point 7.', "govt shouldn't protect environment,"),
    'V201302x': ('PRE: FEDERAL BUDGET SPENDING: SOCIAL SECURITY', 'Should federal spending on Social Security be increased, decreased, or kept the same?', "decrease social security spending,"),
    'V201305x': ('PRE: SUMMARY: FEDERAL BUDGET SPENDING: PUBLIC SCHOOLS', 'Should federal spending on public schools be increased, decreased, or kept the same?', "decrease public school spending,"),
    'V201308x': ('PRE: SUMMARY: FEDERAL BUDGET SPENDING: TIGHTENING BORDER SECURITY', 'Should federal spending on tightening border security to prevent illegal immigration be increased, decreased, or kept the same?', "decrease border security spending,"),
    'V201311x': ('PRE: SUMMARY: FEDERAL BUDGET SPENDING: DEALING WITH CRIME', 'Should federal spending on dealing with crime be increased, decreased, or kept the same?', "decrease crime prevention spending,"),
    'V201314x': ('PRE: SUMMARY: FEDERAL BUDGET SPENDING: WELFARE PROGRAMS', 'Should federal spending on welfare programs be increased, decreased, or kept the same?', "decrease welfare spending,"),
    'V201317x': ('PRE: SUMMARY: FEDERAL BUDGET SPENDING: BUILDING AND REPAIRING HIGHWAYS', 'Should federal spending on building and repairing highways be increased, decreased, or kept the same?', "decrease highway spending,"),
    'V201320x': ('PRE: SUMMARY: FEDERAL BUDGET SPENDING: AID TO THE POOR', 'Should federal spending on aid to the poor be increased, decreased, or kept the same?', "decrease poor aid spending,"),
    'V201323x': ('PRE: SUMMARY: FEDERAL BUDGET SPENDING: PROTECTING THE ENVIRONMENT', 'Should federal spending on protecting the environment be increased, decreased, or kept the same?', "decrease environment spending,"),
    'V201336': ('RE: STD ABORTION: SELF-PLACEMENT', '1. By law, abortion should never be permitted 2. The law should permit abortion only in case of rape, incest, or when the woman’s life is in danger 3. The law should permit abortion other than for rape/incest/danger to woman but only after need clearly established 4. By law, a woman should always be able to obtain an abortion as a matter of personal choice', "abortion always allowed,"),
    'V201345x': ('PRE: SUMMARY: R FAVOR/OPPOSE DEATH PENALTY', 'Do you favor or oppose the death penalty for persons convicted of murder?', "no death penalty,"),
    'V201349x': ('PRE: SUMMARY: COUNTRY WOULD BE BETTER OFF IF WE JUST STAYED HOME', '‘This country would be better off if we just stayed home and did not concern ourselves with problems in other parts of the world.’', "us should not stay home,"),
    'V201350': ('PRE: FORCE TO SOLVE INTERNATIONAL PROBLEMS', 'How willing should the United States be to use military force to solve international problems?', "us shouldnt use force internationally,"),
    'V201356x': ('PRE: SUMMARY: FAVOR/OPPOSE VOTE BY MAIL', 'Do you favor, oppose, or neither favor nor oppose conducting all elections by mail, instead of people voting in-person?', "vote in person,"),
    'V201359x': ('PRE: SUMMARY: FAVOR/OPPOSE REQUIRING ID WHEN VOTING', 'Do you favor, oppose, or neither favor nor oppose requiring all people to show a government issued photo ID when they vote?', "no voter id laws,"),
    'V201362x': ('PRE: SUMMARY: FAVOR/OPPOSE ALLOWING FELONS TO VOTE', 'Do you favor, oppose, or neither favor nor oppose allowing convicted felons to vote once they complete their sentence?', "ex felons should not vote,"),
    'V201366': ('PRE: HOW IMPORTANT THAT NEWS ORGANIZATIONS FREE TO CRITICIZE', 'For the next few items, we would like to know how important you think each one is to the United States maintaining a strong democracy. First, how important is it that news organizations are free to criticize political leaders?', "important that news can criticize politicians,"),
    'V201367': ('PRE: HOW IMPORTANT BRANCHES OF GOVERNMENT KEEP ONE ANOTHER FROM TOO MUCH POWER', 'How important is it that the executive, legislative, and judicial branches of government keep one another from having too much power?', "checks and balances important,"),
    'V201368': ('PRE: HOW IMPORTANT ELECTED OFFICIALS FACE SERIOUS CONSEQUENCES FOR MISCONDUCT', 'How important is it that elected officials face serious consequences if they engage in misconduct?', "important elected officials face consequences for misconduct,"),
    'V201369': ('PRE: HOW IMPORTANT THAT PEOPLE AGREE ON BASIC FACTS', 'How important is it that people agree on basic facts even if they disagree politically?', "important people agree on basic facts,"),
    'V201372x': ('PRE: SUMMARY: HELPFUL/HARMFUL IF PRES DIDN’T HAVE TO WORRY ABOUT CONGRESS/COURTS', 'Would it be helpful, harmful, or neither helpful nor harmful if U.S. presidents could work on the country’s problems without paying attention to what Congress and the courts say?', "bad if us president could bypass courts/congress,"),
    'V201375x': ('PRE: SUMMARY: FAVOR OR OPPOSE RESTRICTING JOURNALIST ACCESS', 'Do you favor, oppose, or neither favor nor oppose elected officials restricting journalists’ access to information about government decision-making?', "do not restrict journalist access to govt info,"),
    'V201376': ('PRE: HOW CONCERNED GOVERNMENT MIGHT UNDERMINE MEDIA', 'How concerned are you that some people in the government today might want to undermine the news media’s ability to serve as a check on governmental power?', "concerned govt want to undermine media,"),
    'V201377': ('PRE: HOW MUCH TRUST IN NEWS MEDIA', 'In general, how much trust and confidence do you have in the news media when it comes to reporting the news fully, accurately, and fairly?', "confident in news media,"),
    'V201378': ('PRE: APPROPRIATE/INAPPROPRIATE PRES ASK FOREIGN COUNTRIES TO INVESTIGATE RIVALS', 'In general, how appropriate or inappropriate is it for the president of the United States to ask leaders of foreign countries to investigate political rivals?', "pres shouldnt ask foreign countries to investigate rivals,"),
    'V201379': ('PRE: PREFER GOVERNMENT OFFICIAL WHO COMPROMISES OR STICKS TO PRINCIPLES', 'Would you prefer a government official who compromises to get things done, or who sticks to their principles no matter what?', "stick to principles and not compromise,"),
    'V201386x': ('PRE: SUMMARY: FAVOR OR OPPOSE HOUSE IMPEACHMENT DECISION', 'Do you favor, oppose, or neither favor nor oppose the U.S. House of Representatives’ decision in December of last year to impeach President Trump?', "against trump impeachment house,"),
    'V201389x': ('PRE: SUMMARY: FAVOR OR OPPOSE SENATE ACQUITTAL DECISION', 'Do you favor, oppose, or neither favor nor oppose the U.S. Senate’s decision in February to acquit President Trump of the impeachment charges and thus let him remain in office?', "against trump impeachment senate,"),
    'V201392x': ('PRE: SUMMARY: FEDERAL GOVERNMENT RESPONSE TO COVID-19', 'Do you think the federal government’s response to the COVID-19 outbreak earlier this year was too quick, too slow, or aboutright?', "covid response too slow,"),
    'V201405x': ('PRE: SUMMARY: REQUIRE EMPLOYERS TO OFFER PAID LEAVE TO PARENTS OF NEW CHILDREN', 'Do you favor, oppose, or neither favor nor oppose requiring employers to offer paid leave to parents of new children?', "no paid parental leave,"),
    'V201408x': ('PRE: SUMMARY: SERVICES TO SAME SEX COUPLES', 'Do you think business owners who provide wedding-related services should be allowed to refuse services to same-sex couples if same-sex marriage violates their religious beliefs, or do you think business owners should be required to provide services regardless of a couple’s sexual orientation?', "business should provide services to same sex couples,"),
    'V201411x': ('PRE: SUMMARY: TRANSGENDER POLICY', 'Should transgender people - that is, people who identify themselves as the sex or gender different from the one they were born as - have to use the bathrooms of the gender they were born as, or should they be allowed to use the bathrooms of their identified gender?', "against trans bathroom bills,"),
    'V201414x': ('PRE: SUMMARY: FAVOR/OPPOSE LAWS PROTECT GAYS LESBIANS AGAINST JOB DISCRIMINATION', 'Do you favor or oppose laws to protect gays and lesbians against job discrimination?', "oppose gay antidiscrimiation law,"),
    'V201415': ('PRE: SHOULD GAY AND LESBIAN COUPLES BE ALLOWED TO ADOPT', 'Do you think gay or lesbian couples should be legally permitted to adopt children?', "gay people shouldnt be allowed to adopt children,"),
    'V201416': ('PRE: R POSITION ON GAY MARRIAGE', 'Should gay people be allowed to get married?', "no gay marriage,"),
    'V201417': ('PRE: US GOVERNMENT POLICY TOWARD UNAUTHORIZED IMMIGRANTS', 'How should US treat undocumented immigrants?', "allow undocumented immigrants to become citizens,"),
    'V201420x': ('PRE: SUMMARY: FAVOR OR OPPOSE ENDING BIRTHRIGHT CITIZENSHIP', 'Some people have proposed that the U.S. Constitution should be changed so that the children of unauthorized immigrants do not automatically get citizenship if they are born in this country. Do you favor, oppose, or neither favor nor oppose this proposal?', "don't end birthright citizenship,"),
    'V201423x': ('PRE: SUMMARY: SHOULD CHILDREN BROUGHT ILLEGALLY BE SENT BACK OR ALLOWED TO STAY', 'What should happen to immigrants who were brought to the U.S. illegally as children and have lived here for at least 10 years and graduated high school here? Should they be sent back where they came from, or should they be allowed to live and work in the United States?', "children of undocumented shouldn't be deported,"),
    'V201426x': ('PRE: SUMMARY: FAVOR OR OPPOSE BUILDING A WALL ON BORDER WITH MEXICO', 'Do you favor, oppose, or neither favor nor oppose building a wall on the U.S. border with Mexico?', "oppose border wall,"),
    'V201429': ('PRE: BEST WAY TO DEAL WITH URBAN UNREST', 'What is the best way to deal with the problem of urban unrest and rioting?', "use force to prevent urban unrest,"),
    'V201626': ('PRE: NEED TO BE MORE SENSITIVE TALKING OR PEOPLE TOO EASILY OFFENDED', 'Some people think that the way people talk needs to change with the times to be more sensitive to people from different backgrounds. Others think that this has already gone too far and many people are just too easily offended. Which is closer to your opinion?', "people too easily offended"),
    'V201639': ('PRE: WOMEN INTERPRET INNOCENT REMARKS AS SEXIST', 'Do many women interpret innocent remarks or acts as being sexist?', "most women don\'t interpret innocent remarks or acts as being sexist."),
    'V201640': ('PRE: WOMEN SEEK TO GAIN POWER BY GETTING CONTROL OVER MEN', 'do women want to gain power by controlling men', "women do not seek to gain power by controlling men"),
    'V202212': ('POST: [STD] PUBLIC OFFICIALS DON’T CARE WHAT PEOPLE THINK', 'do public officials don’t care much what people like you think?', "public officials care what i think"),
    'V202213': ('POST: [STD] HAVE NO SAY ABOUT WHAT GOVERMENT DOES', 'do people like you have say in what the government does?', "i have no say in govt"),
    'V202231x': ('POST: SUMMARY: FAVOR/OPPOSE NEW LIMITS ON IMPORTS', 'Some people have suggested placing new limits on foreign imports in order to protect American jobs. Others say that such limits would raise consumer prices and hurt American exports. Do you favor or oppose placing new limits on imports?', "no new limits on imports"),
    'V202232': ('POST: WHAT SHOULD IMMIGRATION LEVELS BE', 'Do you think the number of immigrants from foreign countries who are permitted to come to the United States to live should be increased a lot, increased a little, left the same as it is now, decreased a little, or decreased a lot?', "less immigrants"),
    'V202236x': ('POST: SUMMARY: FAVOR/OPPOSE ALLOWING REFUGEES TO COME TO US', 'Do you favor, oppose, or neither favor nor oppose allowing refugees who are fleeing war, persecution, or natural disasters in other countries to come to live in the U.S.?', "oppose refugees"),
    'V202242x': ('POST: SUMMARY: FAVOR/OPPOSE PROVIDING PATH TO CITIZESHIP', 'Do you favor, oppose, or neither favor nor oppose providing a path to citizenship for unauthorized immigrants who obey the law, pay a fine, and pass security checks?', "no path to citizenship for undocumented"),
    'V202245x': ('POST: SUMMARY: FAVOR/OPPOSE RETURNING UNAUTHORIZE IMMIGRANTS TO NATIVE COUNTRY', 'Do you favor, oppose, or neither favor nor oppose returning all unauthorized immigrants to their native countries?', "oppose returning undocumented to birth countries"),
    'V202248x': ('POST: SUMMARY: FAVOR/OPPOSE SEPARATING CHILDREN OF DETAINED IMMIGRANTS', 'Do you favor, oppose, or neither favor nor oppose separating the children of detained immigrants, rather than keeping them with their parents in adult detention centers?', "dont separate undocumented kids from families"),
    'V202252x': ('POST: SUMMARY: FAVOR/OPPOSE PREFERENTIAL HIRING/PROMOTION OF BLACKS', 'are you for or against preferential hiring and promotion of blacks?', "no preferential hiring/promotion of blacks"),
    'V202255x': ('POST: SUMMARY: LESS OR MORE GOVERNMENT', 'less government better or more that govt needs to do', "more govt better"),
    'V202259x': ('POST: SUMMARY: FAVOR/OPPOSE GOVERNMENT TRYING TO REDUCE INCOME INEQUALITY', 'Would it be good for society to have more government regulation, about the same amount of regulation as there is now, or less government regulation?', "govt shouldnt reduce income inequality"),
    'V202260': ('POST: SOCIETY SHOULD MAKE SURE EVERYONE HAS EQUAL OPPORTUNITY', 'Should our society do whatever is necessary to make sure that everyone has an equal opportunity to succeed?', "shouldnt make sure everyone has equal opportunity"),
    'V202261': ('POST: WE’D BE BETTER OFF IF WORRIED LESS ABOUT EQUALITY', 'should we be more worried about equality?', "should worry more about equality"),
    'V202262': ('POST: NOT A BIG PROBLEM IF SOME HAVE MORE CHANCE IN LIFE', 'Is it a problem if some have more chances in life?', "problem if some have more chance in life"),
    'V202263': ('POST: IF PEOPLE WERE TREATED MORE FAIRLY WE WOULD HAVE FEWER PROBLEMS', 'Would we have fewer problems if people were treated more fairly?', "wouldnt have fewer problems if people treated more equally"),
    'V202264': ('POST: THE WORLD IS CHANGING & WE SHOULD ADJUST VIEW OF MORAL BEHAVIOR', 'Should we change our views of moral behavior to fit the changing world?', "shouldn\'t adjust morality to changing world"),
    'V202265': ('POST: FEWER PROBLEMS IF THERE WAS MORE EMPHASIS ON TRADITIONAL FAMILY VALUES', 'Would we have fewer problems if more focus on traditional family ties', "wouldnt have less problems if focused more on traditional family ties"),
    'V202270': ('POST: BETTER IF REST OF WORLD MORE LIKE AMERICA', 'Would the world be better if people from other countries were more like Americans?', "wouldnt be better if others more like americans"),
    'V202273x': ('POST: SUMMARY: US BETTER OR WORSE THAN MOST OTHER COUNTRIES', 'Is the US better or worse than other countries', "US worse than other countries"),
    'V202279x': ('POST: SUMMARY: PEOPLE IN RURAL AREAS HAVE TOO MUCH/TOO LITTLE INFLUENCE', 'Compared to people living in cities, do people living in small towns and rural areas have too much influence, too little influence, or about the right amount of influence on government?', "rural people have too little govt influence"),
    'V202282x': ('POST: SUMMARY: PEOPLE IN RURAL AREAS GET TOO MUCH/TOO LITTLE RESPECT', 'Do people living in small towns and rural areas get too much respect, too little respect, or about the right amount of respect from people living in cities?', "rural people get too little respect"),
    'V202290x': ('POST: SUMMARY: BETTER/WORSE IF MAN WORKS AND WOMAN TAKES CARE OF HOME', 'Is it better or worse if men work and women take care of the home', "worse if traditional gender roles"),
    'V202291': ('POST: DO WOMEN DEMANDING EQUALITY SEEK SPECIAL FAVORS', 'When women demand equality these days, how often are they actually seeking special favors?', "women asking for equality dont want special favors"),
    'V202292': ('POST: DO WOMEN COMPLAINING ABOUT DISCRIMINATION CAUSE MORE PROBLEMS', 'When women complain about discrimination, how often do they cause more problems than they solve?', "women complaining about discrimination dont cause more problems"),
    'V202300': ('POST: AGREE/DISAGREE: BLACKS SHOULD WORK THEIR WAY UP WITHOUT SPECIAL FAVORS', 'should blacks work their way up without special favors', 'disagree w "blacks should work their way up without special favors"'),
    'V202302': ('POST: AGREE/DISAGREE: BLACKS HAVE GOTTEN LESS THAN THEY DESERVE', 'Over the past few years, blacks have gotten less than they deserve', "blacks havent gotten less than they deserve"),
    'V202303': ('POST: AGREE/DISAGREE: IF BLACKS TRIED HARDER THEY’D BE AS WELL OFF AS WHITES', 'Would blacks be as well off as whites if they worked hard enough', "blacks dont suffer prejudice because of lack of hard work"),
    'V202304': ('POST: OUR POLITICAL SYSTEM ONLY WORKS FOR INSIDERS WITH MONEY AND POWER', 'Does the political system only work for insiders with money and power?', "politics only works for powerful"),
    'V202305': ('POST: BECAUSE OF RICH AND POWERFUL IT’S DIFFICULT FOR THE REST TO GET AHEAD', 'Do rich and powerful make it harder for everyone else to get ahead?', "powerful make it harder for others to get ahead"),
    'V202308x': ('POST: SUMMARY: TRUST ORDINARY PEOPLE/EXPERTS FOR PUBLIC POLICY', 'When it comes to public policy decisions, whom do you tend to trust more: ordinary people, experts, or trust both the same?', "trust experts more than ordinary people"),
    'V202309': ('POST: HOW MUCH DO PEOPLE NEED HELP FROM EXPERTS TO UNDERSTAND SCIENCE', 'How much do ordinary people need the help of experts to understand complicated things like science and health?', "people need help from experts to understand science"),
    'V202310': ('POST: HOW IMPORTANT SHOULD SCIENCE BE FOR DECISIONS ABOUT COVID-19', 'In general, how important should science be for making government decisions about COVID-19?', "science important for covid decisions"),
    'V202311': ('POST: BUSINESS AND POLITICS CONTROLLED BY FEW POWERFUL PEOPLE', 'Are business and politics controlled by a few powerful people?', "business/politics controlled by powerful few"),
    'V202312': ('POST: MUCH OF WHAT PEOPLE HEAR IN SCHOOLS AND MEDIA ARE LIES BY THOSE IN POWER', 'Is much of the info in school and media lies designed to benefit powerful?', "schools/media lie to benefit powerful few"),
    'V202317': ('POST: HOW MUCH OPPORTUNITY IN AMERICA FOR AVERAGE PERSON TO GET AHEAD', 'How much opportunity is there in America today for the average person to get ahead?', "no opportunity for average person to get ahead"),
    'V202321': ('POST: IMPORTANCE OF REDUCING DEFICIT', 'How important is it to reduce the deficit?', "reducing defecit not important"),
    'V202328x': ('POST: SUMMARY: APPROVE/DISAPPROVE AFFORDABLE CARE ACT', 'Do you approve, disapprove, or neither approve nor disapprove of the Affordable Care Act of 2010, sometimes called Obamacare?', "disapprove obamacare"),
    'V202331x': ('POST: SUMMARY: FAVOR/OPPOSE REQUIRING VACCINES IN SCHOOLS', 'Do you favor, oppose, or neither favor nor oppose requiring children to be vaccinated in order to attend public schools?', "disapprove vaccines"),
    'V202336x': ('POST: SUMMARY: FAVOR/OPPOSE INCREASED REGULATION ON GREENHOUSE EMISSIONS', 'Do you favor, oppose, or neither favor nor oppose increased government regulation on businesses that produce a great deal of greenhouse emissions linked to climate change?', "oppose climate change regulation"),
    'V202341x': ('POST: SUMMARY: FAVOR/OPPOSE BACKGROUND CHECKS FOR GUN PUCHASES', 'Do you favor, oppose, or neither favor nor oppose requiring background checks for gun purchases at gun shows or other private sales?', "oppose gun purchase background checks"),
    'V202344x': ('POST: SUMMARY: FAVOR/OPPOSE BANNING ‘ASSAULT-STYLE’ RIFLES', 'Do you favor, oppose, or neither favor nor oppose banning the sale of semi-automatic “assault-style” rifles?', "oppose assault weapons ban"),
    'V202347x': ('POST: FAVOR/OPPOSE GOVERNMENT BUY BACK OF ‘ASSAULT-STYLE’ RIFLES', 'Do you favor, oppose, or neither favor nor oppose a mandatory program where the government would buy back semi-automatic assault-style rifles from citizens who currently own them?', "oppose assault weapon buyback"),
    'V202350x': ('POST: SUMMARY SHOULD FEDERAL GOVT DO MORE/LESS ABOUT OPIOID DRUG ADDICTION', 'Do you think the federal government should be doing more about the opioid drug addiction issue, should be doing less, or is it currently doing the right amount?', "govt should do less abt opioid crisis"),
    'V202351': ('POST: HOW OFTEN DO POLICE OFFICERS USE MORE FORCE THAN NECESSARY', 'How often do you think police officers use more force than is necessary?', "police overuse force"),
    'V202364x': ('POST: SUMMARY: INCREASING TRADE GOOD/BAD FOR INTERNATIONAL RELATIONSHIPS', 'Is increasing the amount of international trade good, bad, or neither good nor bad for our relationships with other countries?', "int trade increase bad"),
    'V202373x': ('POST: SUMMARY: INCREASING DIVERSITY MADE US BETTER/WORSE PLACE TO LIVE', 'Does the increasing number of people of many different races and ethnic groups in the United States make this country a better place to live, a worse place to live, or does it make no difference?', "diversity makes US worse"),
    'V202376x': ('POST: SUMMARY: FAVOR/OPPOSE FEDERAL PROGRAM GIVING CITIZENS $12K/YEAR', 'Do you favor, oppose, or neither favor nor oppose establishing a federal program that gives all citizens $12,000 per year, provided they meet certain conditions? This program would be paid for with higher taxes.', "against $12k/year ubi"),
    'V202377': ('POST: SHOULD THE MINIMUM WAGE BE RAISED, KEPT THE SAME, OR LOWERED', 'Should the federal minimum wage be raised, kept the same, lowered but not eliminated, or eliminated altogether?', "decrease minimum wage"),
    'V202380x': ('POST: SUMMARY: INCREASE/DECREASE GOVERNMENT SPENDING TO HELP PAY FOR HEALTH CARE', 'Do you favor an increase, decrease, or no change in government spending to help people pay for health insurance when they can’t pay for it all themselves?', "decrease govt healthcare spending"),
    'V202387x': ('POST: SUMMARY: ATTENTION TO SEXUAL HARRASSMENT AS GONE TOO FAR/NOT FAR ENOUGH', 'Do you think attention to sexual harassment has gone too far, has not gone far enough, or has been about right?', "too little attn. to sexual harassement"),
    'V202390x': ('POST: SUMMARY: FAVOR/OPPOSE TRANSENDER PEOPLE SERVE IN MILITARY', 'Do you favor, oppose, or neither favor nor oppose allowing transgender people to serve in the United States Armed Forces?', "against trans military service"),
    'V202400': ('POST: HOW MUCH IS CHINA A THREAT TO THE UNITED STATES', 'How much is China a threat to the United States?', "China threat to US"),
    'V202401': ('POST: HOW MUCH IS RUSSIA A THREAT TO THE UNITED STATES', 'How much is Russia a threat to the United States?', "Russia threat to US"),
    'V202402': ('POST: HOW MUCH IS MEXICO A THREAT TO THE UNITED STATES', 'How much is Mexico a threat to the United States?', "Mexico threat to US"),
    'V202403': ('POST: HOW MUCH IS IRAN A THREAT TO THE UNITED STATES', 'How much is Iran a threat to the United States?', "Iran threat to US"),
    'V202404': ('POST: HOW MUCH IS JAPAN A THREAT TO THE UNITED STATES', 'How much is Japan a threat to the United States?', "Japan threat to US"),
    'V202405': ('POST: HOW MUCH IS GERMANY A THREAT TO THE UNITED STATES', 'How much is Germany a threat to the United States?', "Germany threat to US"),
    'V202409': ('POST: CSES5-Q04A: ATTITUDES ABOUT ELITES: COMPROMISE IN POLITICS IS SELLING OUT', 'Do you agree with the following statement: "What people call compromise in politics is really just selling out on one’s principles." ', "compromise in politics not selling out on principles."),
    'V202410': ('POST: CSES5-Q04B: ATTITUDES ABOUT ELITES: POLITICIANS DO NOT CARE ABOUT PEOPLE', 'Do you agree with the following statement: "Most politicians do not care about the people." ', "politicians care about people"),
    'V202411': ('POST: CSES5-Q04C: ATTITUDES ABOUT ELITES: MOST POLITICIANS ARE TRUSTWORTHY', 'Do you agree with the following statement: "Most politicians are trustworthy." ', "politicians not trustworthy"),
    'V202412': ('POST: CSES5-Q04D: ATTITUDES ABOUT ELITES: POLITICIANS ARE MAIN PROBLEM IN US', 'Do you agree with the following statement: "Politicians are the main problem in the United States." ', "politicians not main problem in US"),
    'V202413': ('POST: CSES5-Q04E: ATTITUDES ABOUT ELITES: STRONG LEADER IN GOVERNMENT IS GOOD', 'Do you agree with the following statement: "Having a strong leader in government is good for the United States even if the leader bends the rules to get things done." ', "strong leader that bends rule bad for US"),
    'V202414': ('POST: CSES5-Q04F: ATTITUDES ABOUT ELITES: PEOPLE SHOULD MAKE POLICY DECISIONS', 'Do you agree with the following statement: "The people, and not politicians, should make our most important policy decisions." ', "politicans should make policy instead of people"),
    'V202415': ('POST: CSES5-Q04G: ATTITUDES ABOUT ELITES: POLITICIANS ONLY CARE ABOUT THE RICH', 'Do you agree with the following statement: "Most politicians care only about the interests of the rich and powerful." ', "politicians don\'t only care about rich/powerful"),
    'V202416': ('POST: CSES5-Q05A: OUT-GROUP ATTITUDES: MINORITIES SHOULD ADAPT', 'Do you agree with the following statement: "Minorities should adapt to the customs and traditions of the United States." ', "minorities shouldn\'t adapt to customs of US"),
    'V202417': ('POST: CSES5-Q05B: OUT-GROUP ATTITUDES: WILL OF MAJORITY SHOULD ALWAYS PREVAIL', 'Do you agree with the following statement: "The will of the majority should always prevail, even over the rights of minorities." ', "will of majority shouldn\'t prevail over minority rights'")
}

# IDEOLOGY QUESTIONS DATA CLEANING

ideo_data = data.filter(ideo_questions.keys())
# filters ideological questions
question_names = {key: value[0] for key, value in ideo_questions.items()}
question_text = {key: value[1] for key, value in ideo_questions.items()}
positive_answer = {key: value[2] for key, value in ideo_questions.items()}
ideo_data = ideo_data.rename(columns=positive_answer)

for col in ideo_data:
    ideo_data[col] = pd.to_numeric(
        ideo_data[col], errors='coerce')  # reformats data


def weighted_mean_and_std(vals, weights):
    mean = np.average(vals, weights=weights)
    var = np.average((vals-mean)**2, weights=weights)
    sum_weights = np.sum(weights)
    return (mean, sqrt(var * (sum_weights)/(sum_weights - 1.)))

# normalizes answers by z-scores and scales by weight

# i am going to treat null values as 0s after normalization
# since if someone didn't respond we can assume they are roughly neutral on the issue


means = []
stdevs = []

for col_index, col in enumerate(ideo_data):
    col_nulls = np.array(
        [x >= 0 and x < 90 for x in ideo_data[col]]).astype(bool)
    # get indices of not missing data
    min_, max_ = np.amin(ideo_data[col][col_nulls]
                         ), np.amax(weights[col_nulls])
    ideo_data[col] = ideo_data[col].apply(
        lambda x: (x - min_)/(max_ - min_) * 6 + 1)
    # normalize range
    mean, stdev = weighted_mean_and_std(
        ideo_data[col][col_nulls], weights[col_nulls])
    means.append(mean)
    # stdev is innacurate since it doesnt count missing vals
    ideo_data[col] = ideo_data[col].apply(lambda x: (x - mean))
    # we assume that missing vals are someone with no opinion
    ideo_data[col] = ideo_data[col].mul(col_nulls.astype(int))
    mean, stdev = weighted_mean_and_std(
        ideo_data[col][col_nulls], weights[col_nulls])
    ideo_data[col] = ideo_data[col].apply(lambda x: (x/stdev))
    stdevs.append(stdev)

# DEMOGRAPHIC QUESTIONS DATA CLEANING

demographic_vars = {'V201435': 'PRE: WHAT IS PRESENT RELIGION OF R',
                    'V201462': 'PRE: RELIGIOUS IDENTIFICATION',
                    'V201508': 'PRE: MARITAL STATUS',
                    'V201511x': 'PRE: SUMMARY: RESPONDENT 5 CATEGORY LEVEL OF EDUCATION',
                    'V201601': 'PRE: SEXUAL ORIENTATION OF R [REVISED]',
                    'V201617x': 'PRE: SUMMARY: TOTAL (FAMILY) INCOME',
                    'V202110x': 'PRE-POST: SUMMARY: 2020 PRESIDENTIAL VOTE',
                    'V201600': 'PRE: WHAT IS YOUR (R) SEX? [REVISED]',
                    'V201549x': 'PRE: SUMMARY: R SELF-IDENTIFIED RACE/ETHNICITY',
                    'V201021': 'PRE: FOR WHICH CANDIDATE DID R VOTE IN PRESIDENTIAL PRIMARY',
                    'V201200': 'PRE: 7PT SCALE LIBERAL-CONSERVATIVE SELFPLACEMENT',
                    'V201507x': 'Age'}

# meaning of each value in a demographic variable
# value 0 in original data corresponds to value at index 0 here,
demographic_values = {'V201435': {1: 'Protestant', 2: 'Catholic', 3: 'Orthodox', 4: 'Mormon', 5: 'Jewish',
                                  6: 'Muslim', 7: 'Buddhist', 8: 'Hindu', 9: 'Atheist', 10: 'Agnostic', 11: 'Something else', 12: 'Nothing in particular'},
                      'V201462': {2: 'Charismatic/Pentecostal', 5:  'Traditional', 6: 'Mainline', 7: 'Progressive', 8:  'Non-traditional believer', 9:  'Secular', 12:  'Spiritual but not religious', 13: 'None of the above'},
                      'V201508': {1: 'Married', 2: 'Married', 3: 'Widowed', 4: 'Divorced', 5: 'Separated', 6: 'Never married'},
                      'V201511x': {1:  'Less than high school credential', 2: 'High school credential', 3: 'Some post-high school, no bachelor’s degree', 4: 'Bachelor’s degree', 5: 'Graduate degree'},
                      'V201601': {1: 'Heterosexual or straight', 2: 'Homosexual or gay (or lesbian)', 3: 'Bisexual', 4: 'Something else'},
                      'V202110x': {1: 'Joe Biden', 2: 'Donald Trump', 3: 'Jo Jorgensen', 4: 'Howie Hawkins', 5: 'Other'},
                      'V201600': {1: 'Male', 2: 'Female'},
                      'V201549x': {1: 'White', 2: 'Black', 3: 'Hispanic', 4: 'Asian/NHPI', 5: 'AIAN', 6: 'Multiracial'},
                      'V201021': {1: 'Joe Biden', 2: 'Michael Bloomberg', 3: 'Pete Buttigieg', 4: 'Amy Klobuchar', 5: 'Bernie Sanders', 6: 'Elizabeth Warren', 7: 'Another Democrat', 8: 'Donald Trump', 9: 'Another Republican', 10: '3rd party'},
                      'V201200': {1: 'Extremely liberal', 2: 'Liberal', 3: 'Slightly liberal', 4: 'Moderate; middle of the road', 5: 'Slightly conservative', 6: 'Conservative', 7: 'Extremely conservative'},
                      'V201617x': {1: "Under $9,999", 2: "$10,000-14,999", 3: "$15,000-19,999", 4: "$20,000-24,999", 5: "$25,000-29,999", 6: "$30,000-34,999", 7: "$35,000-39,999", 8: "$40,000-44,999", 9: "$45,000-49,999", 10: "$50,000-59,999", 11: "$60,000-64,999", 12: "$65,000-69,999", 13: "$70,000-74,999", 14: "$75,000-79,999", 15: "$80,000-89,999", 16: "$90,000-99,999", 17: "$100,000-109,999", 18: "$110,000-124,999", 19: "$125,000-149,999", 20: "$150,000-174,999", 21: "$175,000-249,999", 22: "$250,000 or more"}}

demo_data = data.filter(demographic_vars.keys())


demo_data.replace(demographic_values, inplace=True)

for col in demographic_values:
    demo_data[col] = demo_data[col].mask(
        pd.to_numeric(demo_data[col], errors='coerce').notna())

demo_data = demo_data.rename(columns=demographic_vars)

ideo_data = pd.concat([ideo_data, weights], axis=1)
demo_data = pd.concat([demo_data, weights], axis=1)

ideo_data = ideo_data.rename(columns={'V200010b': 'weights'})
demo_data = demo_data.rename(columns={'V200010b': 'weights'})

ideo_data.to_csv(path_to_repo + 'ideo.csv')
demo_data.to_csv(path_to_repo + 'demo.csv')
