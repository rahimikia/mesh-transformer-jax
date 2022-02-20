import argparse
import json
import time
import jax
import numpy as np
import optax
from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt, write_ckpt
from mesh_transformer.transformer_shard import CausalTransformer
from smart_open import open
from google.cloud import storage
from mesh_transformer.util import clip_by_global_norm, additive_weight_decay, to_bf16, to_f16



def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default=None, help="Config file location")
    
    
    parser.add_argument("--ckpt-step", type=int, default=-1, help="Step number of the checkpoint to convert (if not specified, converts the most recent checkpoint)")
    parser.add_argument("--f16", default=False, action="store_true", help="Convert to float16 (instead of bfloat16)")
    
    parser.add_argument("--ticker_name", type=str, default = False, help="Ticker name")
    parser.add_argument("--year", type=str, default = False, help="Year")
    
    args = parser.parse_args()
    return args

args_fx = parse_args()
ticker = args_fx.ticker_name
year = args_fx.year

### Model config (from json file).
txt_train_path = 'gs://nlp-project0/data/GPTJ_data/' + ticker + '/headlines_train_tf/news_headlines_train_' + str(year) + '.tfrecords'
model_name = ticker + '_' + str(year)

 
# GPT-J properties.
storage_client_x = storage.Client()
bucket_x = storage_client_x.get_bucket('nlp-project0')
blobs_x = bucket_x.get_blob('data/GPTJ_data/' + ticker + '/headlines_train_tf/news_headlines_train_' + str(year) + '.txt')
total_steps_val = int(np.round(int(blobs_x.download_as_string())/8))
warmup_steps_val = int(np.round(0.1*total_steps_val))
anneal_steps_val = total_steps_val - warmup_steps_val
 


params = {
  "layers": 28,
  "d_model": 4096,
  "n_heads": 16,
  "n_vocab": 50400,
  "norm": "layernorm",
  "pe": "rotary",
  "pe_rotary_dims": 64,

  "seq": 2048,
  "cores_per_replica": 8,
  "per_replica_batch": 1,
  "gradient_accumulation_steps": 8,

  "warmup_steps": warmup_steps_val,
  "anneal_steps": anneal_steps_val,
  "lr": 5e-5,
  "end_lr": 1e-5,
  "weight_decay": 0.1,
  "total_steps": total_steps_val,

  "tpu_size": 8,

  "bucket": "nlp-project0",
  "model_dir": "finetuned_models_stage_1/" + ticker + '/' + str(year),

  "train_set": "foo.train.index",
  "val_set": {},

  "eval_harness_tasks": [
  ],

  "val_batches": 0,
  "val_every": total_steps_val+100,
  "ckpt_every": total_steps_val,
  "keep_every": total_steps_val,

  "name": model_name,
  "wandb_project": "mesh-transformer-jax",
  "comment": ""
}






if __name__ == "__main__":
    args = parse_args()
    # params = json.load(open(args.config))
    convert_fn = to_f16 if args.f16 else to_bf16

    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    bucket = params["bucket"]
    model_dir = params["model_dir"]

    params["optimizer"] = optax.chain(
        optax.scale(1),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.additive_weight_decay(0),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
    )

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    with open(f"gs://{bucket}/{model_dir}/meta.json", "r") as f:
        meta = json.load(f)

    if args.ckpt_step > -1:
        ckpt_step = args.ckpt_step
    else:
        ckpt_step = meta["checkpoints"][-1]
    print(f"using checkpoint {ckpt_step}")

    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
        network = CausalTransformer(params)

        start = time.time()
        network.state = read_ckpt(network.state, f"gs://{bucket}/{model_dir}/step_{ckpt_step}/", devices.shape[1])
        print(f"network loaded in {time.time() - start:.06}s")

        start = time.time()
        del network.state["opt_state"]

        network.state["params"] = convert_fn(network.state["params"])
        print(f"network converted in {time.time() - start:.06}s")

        suffix = "_slim_f16" if args.f16 else "_slim"

        for i in range(cores_per_replica):
            write_ckpt(network.state, f"gs://{bucket}/{model_dir}{suffix}/step_{ckpt_step}/", i)
            print(f"written shard {i}")

 
## Cleaning cloud (1)

storage_client_1 = storage.Client()
bucket_1 = storage_client_1.get_bucket('nlp-project0')
blobs_1 = bucket_1.list_blobs(prefix = "finetuned_models_stage_1/" + ticker + '/' + str(year) + '/')

for blob in blobs_1:
  blob.delete()

exit()
