import pandas as pd

# Эталонные значения категорий, зафиксированные на этапе обучения.
# Гарантируют одинаковый набор признаков для любых новых данных.
WEBSITE_CATEGORIES = [f'Category {i:02d}' for i in range(1, 21)]
DAYTIME_CATEGORIES = ['утро', 'день', 'вечер', 'ночь']


def build_features(visits, surf_depth, primary_device, cloud_usage, users=None,
                   website_categories=WEBSITE_CATEGORIES,
                   daytime_categories=DAYTIME_CATEGORIES):
    """
    Собирает единое признаковое пространство на уровне пользователя.

    Параметры:
        visits, surf_depth, primary_device, cloud_usage — сырые таблицы.
        users — необязательная таблица с целевой переменной age_category.
        website_categories, daytime_categories — эталонные списки категорий,
            гарантирующие одинаковый набор признаков на новых данных.

    Возвращает:
        DataFrame, индексированный по user_id, с фиксированным набором признаков.
    """
    # 1. Доли активности по категориям сайтов
    cat_share = pd.crosstab(visits['user_id'], visits['website_category'], normalize='index')
    cat_share = cat_share.reindex(columns=website_categories, fill_value=0)
    cat_share = cat_share.add_prefix('cat_')

    # 2. Доли активности по времени суток
    daytime_share = pd.crosstab(visits['user_id'], visits['daytime'], normalize='index')
    daytime_share = daytime_share.reindex(columns=daytime_categories, fill_value=0)

    # 3. Общая активность
    grouped = visits.groupby('user_id')
    activity = pd.DataFrame({
        'total_visits':        grouped.size(),
        'active_days':         grouped['date'].nunique(),
        'n_unique_categories': grouped['website_category'].nunique(),
        'n_sessions':          grouped['session_id'].nunique(),
    })
    activity['sessions_per_day'] = activity['n_sessions'] / activity['active_days']
    activity = activity.drop(columns='n_sessions')

    # 4. Объединяем признаки из логов
    features = cat_share.join(daytime_share).join(activity)

    # 5. Присоединяем пользовательские таблицы (категориальные)
    features = features.join(surf_depth.set_index('user_id'))
    features = features.join(primary_device.set_index('user_id'))
    features = features.join(cloud_usage.set_index('user_id'))

    # cloud_usage -> строка; пропуски -> 'unknown'
    features['cloud_usage'] = features['cloud_usage'].astype(str)
    features['surf_depth'] = features['surf_depth'].fillna('unknown')
    features['primary_device'] = features['primary_device'].fillna('unknown')
    features['cloud_usage'] = features['cloud_usage'].replace('nan', 'unknown')

    # удаляем коллинеарный признак (φK ≈ 1.0 с total_visits)
    features = features.drop(columns='sessions_per_day')

    # 6. Опционально присоединяем целевую переменную
    if users is not None:
        features = features.join(users.set_index('user_id'))

    return features
