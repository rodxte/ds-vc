def read_abt(x, y):
      
    # READ OFFLINE MULTIPLE GOAL FROM META FILE
    offline_goal = list(meta['id'][meta['name'].str.contains('offline')])[0]
    offline_goal = 'goal_{}_converted'.format(offline_goal)
    
    # READ MAIN FILE
    df = pd.read_csv(x, parse_dates=['hit_time'], compression='gzip')
    df.columns = df.columns\
        .str.strip()\
        .str.lower()\
        .str.replace(' ', '_')\
        .str.replace('(', '')\
        .str.replace(')', '')

    # Return jurisdiction
    df['jurisdiction'] = df['custom_dimension'].str.extract(r'jurisdiction=([a-zA-Z]{2})')
    
    # Return VWO_UUID
    df['uuid'] = df['custom_dimension'].str.extract(r'vwo_uuid=([a-zA-Z0-9]{32,33}|(vwo_anonymized))').get(0)
    
    # Return User_id
    df['user_id'] = df['custom_dimension'].str.extract(r'user_id=([a-zA-Z0-9]{31,33})').get(0)

    # Check whether user has duplicates
    df['has_duplicates'] = df[df['user_id'].notnull()].duplicated(subset=['user_id'], keep=False)
    
    # Remove NaNs from goals and make them either 0, or 1
    # Convert default value 1 to int
    goals_time = df.columns[df.columns.str.contains('_converted') & ~df.columns.str.endswith('_time')]
    df[goals_time] = df[goals_time].fillna(0).astype(int)


    # Strip the second goal hit
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
    
    # Convert to booleans Converted Column
    df['returning_visitor'] = df['returning_visitor'].map(
                   {True: 'returning_user', False: 'new_user'})
    
    # Mask goals with value ==2
    # DO NOT MASK OFFLINE GOALS!!!
    goals_converted = \
        df.columns[df.columns.str.startswith('goal_')\
               & df.columns.str.endswith('_converted')\
               & df.columns.str.startswith('goal_')\
               & ~df.columns.str.contains(offline_goal)]
    for x in list(goals_converted):
        df[x] = df[x].mask(df[x]>1, 1)
    
    # Extract landing data and landing number if any
    if df['referring_url'].isna().all() == False:
        df['land'] = df['referring_url'].str.contains('land', na=False)
        df['land_number'] =\
        df['referring_url']\
            .str.extract(r'(land[\d]?[\d\d])')
    else:
        pass
    df['domain'] = df['url'].str.split('https://').str.get(-1).str.split('#login').str.get(0).str.split('/').str.get(0)
    
#     Conver Country Names with COCO
#     df['country_code'] = df['country'].apply(cc.convert, to='ISO2', not_found=None)

    df['screen_width'] = df['screen_resolution'].str.split('x').str.get(0).astype(float).astype('Int64')
    df['screen_viewport'] = pd.cut(df['screen_width'], bins=[319, 479, 767, 959, 1023, 1279, 1439, 1919, 9999],\
       labels=['sm', 'xsm', 'md', 'xmd', 'no', 'xno', 'lg', 'xlg'])
    
    
    all_goal_times = df.columns[df.columns.str.contains('_converted') & df.columns.str.contains(r'(([0-9]{1,3}))', regex=True) &df.columns.str.endswith('_time')]
    for z in all_goal_times:
        df[z.lstrip('goal_').rstrip('_converted_time') + '_goal_secs'] = (df[z]-df['hit_time']).dt.total_seconds()
        df[z.lstrip('goal_').rstrip('_converted_time') + '_goal_days'] = (df[z]-df['hit_time'])
    
    df['day'] = df['hit_time'].dt.day_name()
    return df