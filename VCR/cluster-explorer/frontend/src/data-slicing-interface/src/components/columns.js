export const COLUMNS = [
    {
        headerName: 'items',
        children: [
            {
                headerName: 'Itemset',
                field: 'itemsets',
                // type: 'description',
                cellDataType: false,
                editable: true
            },
        ]
    },
    {
        headerName: 'metrics',
        children: [
            {
                headerName: 'Support',
                field: 'support',
                filter: 'agNumberColumnFilter',
                type: 'float',
            },
            {
                headerName: 'Support Count',
                field: 's_count',
                type: 'int'
            },
            {
                headerName: 'Accuracy',
                field: 'accuracy',
                type: 'float'
            },
            {
                headerName: 'd_Accuracy',
                field: 'd_accuracy',
                type: 'float'
            }
        ]
    }
]
