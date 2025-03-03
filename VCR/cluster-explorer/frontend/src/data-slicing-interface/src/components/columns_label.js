export const COLUMNS_LABEL = [
    {
        headerName: 'items',
        children: [
            {
                headerName: 'Ground Truth',
                field: 'gt_label',
            },
            {
                headerName: 'Predicted Label',
                field: 'pred_label',
            },            
            {
                headerName: 'Itemset',
                field: 'itemsets',
                type: 'description',
                hide: true,
            },
            {
                headerName: 'Itemset',
                field: 'no_label_itemsets',
                type: 'description',
            }
        ]
    },
    {
        headerName: 'metrics',
        children: [
            {
                headerName: 'Support',
                field: 'support',
                filter: 'agNumberColumnFilter',
                type: 'float'
            },
            {
                headerName: 'Support Count',
                field: 's_count',
                type: 'int'
            },
            // {
            //     headerName: 'False Positive Count',
            //     field: 'fp_count',
            //     type: 'nu'
            // },
            {
                headerName: 'False Positive Rate',
                field: 'fpr',
                type: 'float'
            }
        ]
    }
]
