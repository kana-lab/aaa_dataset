use csv::ReaderBuilder;
use tch::{Kind, Tensor};

// must contain at least 1 asset
const ASSETS: [&str; 7] = [
    "ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "ETHUSDT", "TRXUSDT", "XRPUSDT",
];
const N_MONTH_TRAIN: usize = 12;
const N_MONTH_TEST: usize = 3;
const WINDOW_SIZE: usize = 50;
const BATCH_SIZE: usize = 50;

fn main() -> anyhow::Result<()> {
    // global price matrix
    let mut gpm = Vec::new();
    let mut gpm_test = Vec::new();

    let mut check_sum: Option<usize> = None;
    let mut check_sum_test: Option<usize> = None;
    for asset in ASSETS {
        let mut check_sum_for_this_asset = 0;
        let mut check_sum_for_this_asset_test = 0;

        for i in 0..(N_MONTH_TRAIN + N_MONTH_TEST) {
            let file_path = format!("./binance/{}/{}.csv", asset, i);
            println!("extracting from {} ... ", file_path);

            // read csv file

            let file = std::fs::File::open(file_path)?;
            let mut csv_reader = ReaderBuilder::new()
                .has_headers(false)
                .from_reader(file);

            for result in csv_reader.records() {
                let record = result?;

                // record format is provided here:
                //     https://www.binance.com/en/landing/data
                // we use close price of K-Line - Spot data
                let close_price = record[4].parse::<f64>()?;
                if i < N_MONTH_TRAIN {
                    gpm.push(close_price);
                    check_sum_for_this_asset += 1;
                } else {
                    gpm_test.push(close_price);
                    check_sum_for_this_asset_test += 1;
                }
            }
        }

        println!(
            "fetched all {} data ({} + {} entry)\n",
            asset, check_sum_for_this_asset, check_sum_for_this_asset_test
        );

        // verify that the total number of CSV records is the same for each asset
        if check_sum.is_none() {
            check_sum = Some(check_sum_for_this_asset);
            check_sum_test = Some(check_sum_for_this_asset_test);
        } else {
            assert_eq!(check_sum_for_this_asset, check_sum.unwrap());
            assert_eq!(check_sum_for_this_asset_test, check_sum_test.unwrap());
        }
    }

    // make global price matrix Tensor
    let gpm = Tensor::cat(&[
        // add the price of the riskless asset (which is always 1)
        Tensor::from_slice(&vec![1.; check_sum.unwrap()]),
        Tensor::from_slice(&gpm)
    ], 0).reshape(&[(ASSETS.len() + 1) as i64, check_sum.unwrap() as i64]);
    let gpm_test = Tensor::cat(&[
        Tensor::from_slice(&vec![1.; check_sum_test.unwrap()]),
        Tensor::from_slice(&gpm_test)
    ], 0).reshape(&[(ASSETS.len() + 1) as i64, check_sum_test.unwrap() as i64]);

    // TEST: confirmed the gpm was correctly constructed
    // println!("{}", gpm);

    // generate local price matrices and price change rates
    let (lpm, pcr) = make_local_price_matrix(gpm, WINDOW_SIZE);
    let (lpm_test, pcr_test) = make_local_price_matrix(gpm_test, WINDOW_SIZE);

    // make mini-batches and price change matrices
    let (lpm_batches, pcr_matrices) = make_mini_batches(lpm, pcr, BATCH_SIZE);
    println!("{}\n{}", lpm_batches, pcr_matrices);

    // write out the training data to .safetensors
    Tensor::write_safetensors(&[
        // this type conversion is IMPORTANT
        ("lpm_batches", lpm_batches.totype(Kind::Float)),
        ("pcr_matrices", pcr_matrices.totype(Kind::Float)),
        ("test_lpm_batch", lpm_test.totype(Kind::Float)),
        ("test_pcr_matrix", pcr_test.totype(Kind::Float))
    ], "./environment.safetensors")?;

    Ok(())
}

// returns local price matrix and price change vectors
fn make_local_price_matrix(gpm: Tensor, window_size: usize) -> (Tensor, Tensor) {
    let n_columns = gpm.size()[1] as usize;
    let mut lpm = Vec::new();
    let mut price_change_rate_t = Vec::new();

    for i in (window_size - 1)..(n_columns - 1) {
        let window = gpm.narrow(
            1, (i + 1 - window_size) as i64, window_size as i64,
        );
        let current_price = gpm.narrow(1, i as i64, 1);
        let next_price = gpm.narrow(1, i as i64 + 1, 1);
        let y = (&next_price / &current_price).flatten(0, -1);
        lpm.push(window / current_price);
        price_change_rate_t.push(y);
    }

    (Tensor::stack(&lpm, 0), Tensor::stack(&price_change_rate_t, 0))
}

#[test]
fn test_make_local_price_matrix() {
    let gpm = Tensor::from_slice(&[
        1., 1., 1., 1., 1., 2., 3., 4., 5., 6., 2., 5., 6., 8., 10.
    ]).reshape(&[3, 5]);
    let (lpm, pcr) = make_local_price_matrix(gpm, 3);
    println!("{}", lpm);
    println!("{}", pcr);
}

// returns an array of mini-batches and corresponding array of price change vectors
fn make_mini_batches(lpm: Tensor, pcr: Tensor, batch_size: usize) -> (Tensor, Tensor) {
    let batch_size = batch_size as i64;
    let (data_len, m, n) = (lpm.size()[0], lpm.size()[1], lpm.size()[2]);
    let adjusted_size = data_len - data_len % batch_size;
    let n_batch = adjusted_size / batch_size;

    let lpm = lpm.narrow(0, 0, adjusted_size);
    let pcr = pcr.narrow(0, 0, adjusted_size);
    (lpm.view([n_batch, batch_size, m, n]),
     pcr.view([n_batch, batch_size, m]))
}

#[test]
fn test_make_mini_batches() {
    let v = &(0..36).collect::<Vec<_>>();
    let lpm = Tensor::from_slice(&v).view([9, 2, 2]);

    let r = &(0..18).collect::<Vec<_>>();
    let pcr = Tensor::from_slice(&r).view([9, 2]);

    println!("{}\n{}", lpm, pcr);

    let (lpm_batch, pcr_batch) = make_mini_batches(lpm, pcr, 4);
    println!("{}\n{}", lpm_batch, pcr_batch);
}


#[test]
fn t() {
    // 2x3の行列
    let matrix = Tensor::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).reshape(&[2, 3]);

    // 要素数2のベクトル
    let vector = Tensor::from_slice(&[2.0, 3.0]).reshape(&[2, 1]);

    // 行列の各要素をベクトルの対応する要素で割る
    let result = matrix / vector;

    // 結果の表示
    println!("{}", result);

    Tensor::write_safetensors(&[("test", result)], "./test.safetensors").unwrap();
}

#[test]
fn t2() {
    let result = Tensor::read_safetensors("./test.safetensors").unwrap();
    let ts = &result[0].1;
    println!("{}", ts);
}

#[test]
fn t3() {
    let v = Tensor::from_slice(&[1, 2, 3]);
    let v2 = Tensor::from_slice(&[4, 5, 6]);
    let ts = Tensor::cat(&[v2, v], 0);
    // let ts = Tensor::stack(&[v2, v], 0);
    println!("{}", ts);
}