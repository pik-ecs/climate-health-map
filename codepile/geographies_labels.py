from climate_health_map.data.geographies import load_country_infos

df_countries = load_country_infos()
print(df_countries.columns)

output = ''
for prefix in ['Location', 'Affiliation']:
    for column in [
        'Group (Lancet 2026)',
        'Group (WHO 2026)',
        'Group (HDI 2026)',
        'Region (IPCC AR6, 6)',
        'Region (IPCC AR6, 10)',
        'Region (WorldBank 2026)',
        'Income group (WorldBank 2026)',
        'Lending category (WorldBank 2026)',
        'Continent (Name)',
    ]:
        key = f'{prefix}_{column}'
        output += f"""
        '{key}': Group(
            collection=Collection.GEO,
            key='{key}',
            name='{prefix} grouped by {column}',
            type='multi',
            colour=(0, 0, 0),
            labels=[
        """
        for _gi, (grouping, count) in enumerate(df_countries[df_countries[column].notna()][column].value_counts().items()):
            print(f'{prefix} -> {column} -> {grouping}: {count}')
            # output += f"""
            #     Label(
            #         column='{key}|{gi}',
            #         value={gi},
            #         name='{grouping}',
            #         desc='Article with at least one {prefix} in {grouping}',
            #         colour=(0, 0, 0),
            #         parent='{key}',
            #     ),
            # """
            output += f"""
                       Label(
                           column='{key}|{grouping}',
                           value='{grouping}',
                           name='{grouping}',
                           desc='Article with at least one {prefix} in {grouping}',
                           colour=(0, 0, 0),
                           parent='{key}',
                       ),
                   """
        output += '],\n),'

        # f'{prefix}|{column}|{grouping}'

    print('--')

print(output)
