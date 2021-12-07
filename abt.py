from pandas import read_csv, cut # read_abt
from pandas import merge, DataFrame, concat
from scipy.stats import norm # p_value
from numpy import abs, inf, nan
from boto3 import client

def read_abt(main, meta):
    # -----------  MAKE PRETTY  -----------
    df = read_csv(main, parse_dates=['hit_time'], compression='gzip')
    df.columns = df.columns\
        .str.strip()\
        .str.lower()\
        .str.replace(' ', '_')\
        .str.replace('(', '')\
        .str.replace(')', '')

    # Convert All String Values Of Object Type To Lowercase    
    no_goals = df.filter(regex='^(?!.*goal.*$)').columns
    no_goals = df[no_goals].select_dtypes(include='object').columns
    for i in list(no_goals):
        df[i] = df[i].str.lower()\
            .str.replace(' ', '_')\
            .str.replace('-', '_')\

    # -----------  WORK WITH COLUMNS  -----------

    df['day'] = df['hit_time'].dt.day_name()

    # Return jurisdiction
    df['jurisdiction'] = df['custom_dimension'].str.extract(r'jurisdiction=([a-zA-Z]{2})')
    
    # Return VWO_UUID
    df['uuid'] = df['custom_dimension'].str.extract(r'vwo_uuid=([a-zA-Z0-9]{32,33}|(vwo_anonymized))').get(0).str.upper()
    
    # Return hashed_user_id - that means project hashed id
    df['hashed_user_id'] = df['custom_dimension'].str.extract(r'user_id=([a-zA-Z0-9]{31,33})').get(0)

    # Check whether user has duplicates
    df['has_duplicates'] = df[df['hashed_user_id'].notnull()].duplicated(subset=['hashed_user_id'], keep=False)

    df['domain'] = df['url'].str.split('https://').str.get(-1).str.split('#login').str.get(0).str.split('/').str.get(0)

    # Extract landing data and landing number if any
    if df['referring_url'].isna().all() == False:
        df['land'] = df['referring_url'].str.contains('land', na=False)
        df['land_number'] =\
        df['referring_url']\
            .str.extract(r'(land[\d]?[\d\d])')
    else:
        pass

    # Conver Country Names with COCO
    # df['country_code'] = df['country'].apply(cc.convert, to='ISO2', not_found=None)

    # df['screen_width'] = df['screen_resolution'].str.split('x').str.get(0).astype(float).astype('Int64')
    # df['screen_viewport'] = cut(df['screen_width'], bins=[319, 479, 767, 959, 1023, 1279, 1439, 1919, 9999],\
    #    labels=['sm', 'xsm', 'md', 'xmd', 'no', 'xno', 'lg', 'xlg'])

    # Convert to booleans Converted Column
    df['returning_visitor'] = df['returning_visitor'].map(
                   {True: 'returning_user', False: 'new_user'})

    
    # -----------  WORK WITH GOALS  -----------

    # READ OFFLINE MULTIPLE GOAL FROM META FILE
    if meta['name'].str.contains('offline').any()==True:
        offline_goal = list(meta['id'][meta['name'].str.contains('offline')])[0]
        offline_goal = 'goal_{}_converted'.format(offline_goal)
    else:
        offline_goal = 0

    # Remove NaNs from goals and make them either 0, or 1
    # Convert default value 1 to int
    goals_time = df.columns[df.columns.str.contains('_converted') & ~df.columns.str.endswith('_time')]
    df[goals_time] = df[goals_time].fillna(0).astype(int)


    # Strip the second goal hit
    # BE CAREFUL WHEN MULTIPLE OFFLINE GOALS ARE USED
    goals_converted_time = df.columns[df.columns.str.startswith('goal_') & df.columns.str.endswith('_time')]
    for x in list(goals_converted_time):
        df[x] = df[x].astype(str).str.split('|').str.get(0)
    
    # Strip Revenue Metrics of $ Sign and Convert Values to Float
    revenue = df.columns[df.columns.str.endswith('_total') | df.columns.str.endswith('_revenue')]
    for y in list(revenue):
        if df[y].isna().all() == False:
            df[y] = df[y].str.split('|')\
                .str.get(0)\
                .str[1:].astype(float)
        else:
            pass

    # Convert all goals to datetime
    df[goals_converted_time] = df[goals_converted_time].astype('datetime64')


    # Convert to booleans Converted Column
    df['converted'] = df['converted'].map(
                   {'Yes': True, 'No': False})
    
    
    # Mask goals with value ==2 - BE CAREFUL WHEN MULTIPLE OFFLINE GOALS ARE USED
    if type(offline_goal) == str:
        goals_converted = df.columns[df.columns.str.startswith('goal_')\
               & df.columns.str.endswith('_converted')\
               & df.columns.str.startswith('goal_')\
               & ~df.columns.str.contains(offline_goal)]
    else:
        goals_converted = df.columns[df.columns.str.startswith('goal_')\
           & df.columns.str.endswith('_converted')\
           & df.columns.str.startswith('goal_')]
            
    for x in list(goals_converted):
        df[x] = df[x].mask(df[x]>1, 1)

    
    all_goal_times = df.columns[df.columns.str.contains('_converted')\
                                & df.columns.str.contains(r'(([0-9]{1,3}))', regex=True)\
                                &df.columns.str.endswith('_time')]
    for z in all_goal_times:
        df[z.lstrip('goal_').rstrip('_converted_time') + '_goal_secs'] = (df[z]-df['hit_time']).dt.total_seconds()
        # df[z.lstrip('goal_').rstrip('_converted_time') + '_goal_days'] = (df[z]-df['hit_time'])
    
    return df

def p_factor(*factor, goal, df, variations=[1,2], min_visitors=100):
    # from scipy.stats import mannwhitneyu, ttest_ind, f_oneway
    if factor:
        comb1 = df[df['combination_id']==variations[0]].groupby([*factor])[goal]\
            .agg(visitors='count', n_of_conversions='sum', cr='mean', sem='sem').reset_index()
        comb1 = comb1[comb1['visitors'] >= min_visitors].reset_index(drop=True)
        comb1 = comb1[comb1['n_of_conversions'] >= 20].reset_index(drop=True)
        
        comb2 = df[df['combination_id']==variations[1]].groupby([*factor])[goal]\
            .agg(visitors='count', n_of_conversions='sum', cr='mean', sem='sem').reset_index()
        comb2 = comb2[comb2['visitors'] >= min_visitors].reset_index(drop=True)
        comb2 = comb2[comb2['n_of_conversions'] >= 20].reset_index(drop=True)

        combs = merge(comb1, comb2, on=[*factor], suffixes=["_A", "_B"])

        combs['absolute_uplift'] = (combs['cr_B']-combs['cr_A'])
        combs['relative_uplift'] = (combs['cr_B']/combs['cr_A']-1)

        # Calculate Zscore based on SEMs
        zscore = combs['absolute_uplift']/(combs['sem_A']**2 + combs['sem_B']**2)**0.5
        
#         # STAT TESTs
#         a1 = df.groupby(['combination_id', *factor])[goal].get_group(variations[0])
#         b1 = df.groupby(['combination_id', *factor])[goal].get_group(variations[1])
#         combs['mannwhitneyu_less'] = round(mannwhitneyu(a1, b1, alternative='less').pvalue,4)
#         combs['ttestind_less'] = round(ttest_ind(a1, b1, alternative='less').pvalue,4)
#         combs['mannwhitneyu_less_2'] = round(mannwhitneyu(a1, b1, alternative='two-sided').pvalue,4)
#         combs['ttestind_less_2'] = round(ttest_ind(a1, b1, alternative='two-sided').pvalue,4)

        # Calculate P-Values out of Zscore
        combs['p-value'] = norm.sf(abs(zscore))
        
        combs['visitors_A'], combs['visitors_B'], combs['n_of_conversions_A'], combs['n_of_conversions_B'] =\
        combs['visitors_A'].astype(int), combs['visitors_B'].astype(int),\
        combs['n_of_conversions_A'].astype(int), combs['n_of_conversions_B'].astype(int)

        # Drop Invalid Values when devided by ZERO
        combs = combs.replace([inf, -inf], nan)
        combs = combs.dropna()

        # Return only Meaningful Columns
        combs = combs.sort_values(by='relative_uplift', ascending=False).reset_index(drop=True)
        combs = combs[[*factor, 'visitors_A', 'visitors_B', 'n_of_conversions_A', 'n_of_conversions_B', 'cr_A', 'cr_B', 'relative_uplift', 'p-value',\
                       'absolute_uplift']]
        combs[['cr_A', 'cr_B', 'relative_uplift', 'absolute_uplift']] = round((combs[['cr_A', 'cr_B', 'relative_uplift', 'absolute_uplift']]*100), 2)
        combs['p-value'] = round(combs['p-value'], 3)

        return combs
    else:
        comb1 = DataFrame(df[df['combination_id']==variations[0]][goal]\
            .agg(visitors='count', n_of_conversions='sum', cr='mean', sem='sem')).T.reset_index(drop=True)
        comb2 = DataFrame(df[df['combination_id']==variations[1]][goal]\
            .agg(visitors='count', n_of_conversions='sum', cr='mean', sem='sem')).T.reset_index(drop=True)
        combs = merge(comb1, comb2, left_index=True, right_index=True, suffixes=["_A", "_B"])
        combs['absolute_uplift'] = (combs['cr_B']-combs['cr_A'])
        combs['relative_uplift'] = (combs['cr_B']/combs['cr_A']-1)

        # Calculate Zscore based on SEMs
        zscore = combs['absolute_uplift']/(combs['sem_A']**2 + combs['sem_B']**2)**0.5

        # Calculate P-Values out of Zscore
        combs['p-value'] = norm.sf(abs(zscore))
        
#         # STAT TESTs
#         a1 = df.groupby('combination_id')[goal].get_group(variations[0])
#         b1 = df.groupby('combination_id')[goal].get_group(variations[1])
#         combs['mannwh_less'] = round(mannwhitneyu(a1, b1, alternative='less').pvalue,4)
#         combs['tt_less'] = round(ttest_ind(a1, b1, alternative='less').pvalue,4)
#         combs['mannwh_less_2'] = round(mannwhitneyu(a1, b1, alternative='two-sided').pvalue,4)
#         combs['tt_less_2'] = round(ttest_ind(a1, b1, alternative='two-sided').pvalue,4)
        
        combs['visitors_A'], combs['visitors_B'], combs['n_of_conversions_A'], combs['n_of_conversions_B'] =\
        combs['visitors_A'].astype(int), combs['visitors_B'].astype(int),\
        combs['n_of_conversions_A'].astype(int), combs['n_of_conversions_B'].astype(int)
        
        combs = combs[[*factor, 'visitors_A', 'visitors_B', 'n_of_conversions_A', 'n_of_conversions_B', 'cr_A', 'cr_B', 'relative_uplift', 'p-value',\
               'absolute_uplift']]
        combs[['cr_A', 'cr_B', 'relative_uplift', 'absolute_uplift']] = round((combs[['cr_A', 'cr_B', 'relative_uplift', 'absolute_uplift']]*100),2)
        combs['p-value'] = round(combs['p-value'], 4)
        combs['factor'] = 'all_population'
        first_column = combs.pop('factor')
        combs.insert(0, 'factor', first_column)
        return combs

# def p_goal(goal, df, df_factors):
#     factor_p = df_factors.columns
#     newlist = []
#     newlist.append(p_factor(goal='goal_{}_converted'.format(goal), df=df))
#     for x in range(len(factor_p)):
#         newlist.append(p_factor(factor_p[x], goal='goal_{}_converted'.format(goal), df=df).rename(columns={factor_p[x]: 'factor'}))

#     result = concat(newlist, ignore_index=True)

#     # RETURN SORTED RESULTS
#     return result.sort_values(by=['relative_uplift'], ascending=False)


def tta(*factor, goal, df, variations=[1,2], min=50):
    if factor:
        df = df[abs(df[goal] - df[goal].mean()) <= (3*df[goal].std())]
        comb1 = df[df['combination_id']==variations[0]].groupby([*factor])[goal]\
            .describe().reset_index()
        comb1 = comb1[comb1['count'] >= min].reset_index(drop=True)
        comb2 = df[df['combination_id']==variations[1]].groupby([*factor])[goal]\
            .describe().reset_index() 
        comb2 = comb2[comb2['count'] >= min].reset_index(drop=True)
        
        combs = merge(comb1, comb2, on=[*factor], suffixes=["_A", "_B"])
        combs['count_A'] = combs['count_A'].astype(int)
        combs['count_B'] = combs['count_B'].astype(int)
        # combs['count_C'] = combs['count_C'].astype(int)
        combs = combs[[*factor, 'count_A', 'count_B', '50%_A', '50%_B', '75%_A', '75%_B']]
        return combs
    else:
        df = df[abs(df[goal] - df[goal].mean()) <= (3*df[goal].std())]
        combs = df.groupby('combination_id')[goal].describe()
        return combs