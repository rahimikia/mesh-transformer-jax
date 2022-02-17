import argparse
import json
import time
import jax
import numpy as np
import optax
from tqdm import tqdm
from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt, write_ckpt
from mesh_transformer.transformer_shard import CausalTransformer
from tfrecord_loader import TFRecordNewInputs
from smart_open import open
from google.cloud import storage
from google.cloud.exceptions import NotFound
from mesh_transformer.util import clip_by_global_norm, additive_weight_decay, to_bf16, to_f16

import io
import multiprocessing
import warnings
import os
import re
from typing import Iterable, List, Union
import jax.numpy as jnp
from jax.experimental import maps
from pathy import FluidPath, Pathy
import torch

# TICKERS = ['AAPL', 'GE', 'IBM', 'INTC']
# TICKERS = ['MSFT', 'PFE','PG', 'WMT']
# TICKERS = ['CHV_CVX', 'CSCO', 'BAC', 'CMB_JPM']
# TICKERS = ['KO', 'WFC', 'BRK', 'HON']
# TICKERS = ['MRK', 'BEL_VZ', 'AIG', 'PEP']
# TICKERS = ['JNJ', 'XOM', 'ORCL', 'DIS']


# TICKERS = ['AMGN', 'AMZN', 'AXP', 'WAG_WBA']
# TICKERS = ['BMY', 'BPA_BP', 'CMCSA', 'CVS']
# TICKERS = ['DD', 'GILD', 'GS', 'OXY']
# TICKERS = ['HD', 'MCD', 'MDT', 'MMM', 'BA']
# TICKERS = ['MWD_MS', 'NAN', 'QCOM', 'RY']
# TICKERS = ['SLB', 'TXN', 'UNH', 'UPS', 'USB']


# TICKERS = ['AAPL', 'GE', 'IBM', 'INTC', 'MSFT', 'PFE','PG', 'WMT', 'CHV_CVX', 'CSCO', 'BAC', 'CMB_JPM', 'KO', 'WFC', 'BRK', 'HON']				   
# TICKERS = ['MRK', 'BEL_VZ', 'AIG', 'PEP', 'JNJ', 'XOM', 'ORCL', 'DIS', 'AMGN', 'AMZN', 'AXP', 'WAG_WBA', 'BMY', 'BPA_BP', 'CMCSA', 'CVS']
# TICKERS = ['DD', 'GILD', 'GS', 'OXY', 'HD', 'MCD', 'MDT', 'MMM', 'BA', 'MWD_MS', 'NAN', 'QCOM', 'RY', 'SLB', 'TXN', 'UNH', 'UPS', 'USB']
				
TICKERS = ['AMZN']

								

for ticker in TICKERS:
  print('Transforming:' + ticker)
  for year in np.arange(2005,2006):
    
    ### Model config (from json file).

    txt_train_path = 'gs://nlp-project0/data/GPTJ_data/' + ticker + '/headlines_train_tf/news_headlines_train_' + str(year) + '.tfrecords'
    model_name = ticker + '_' + str(year)
    
    
    ## Modify train test file.
    
    with open("foo.train.index","w") as docA:
        data = docA.write(txt_train_path)
    
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
      "gradient_accumulation_steps": 16,
    
      "warmup_steps": 7,
      "anneal_steps": 65,
      "lr": 5e-5,
      "end_lr": 1e-5,
      "weight_decay": 0.1,
      "total_steps": 72,
    
      "tpu_size": 8,
    
      "bucket": "nlp-project0",
      "model_dir": "finetuned_models_stage_1/" + ticker + '/' + str(year),
    
      "train_set": "foo.train.index",
      "val_set": {},
    
      "eval_harness_tasks": [
      ],
    
      "val_batches": 0,
      "val_every": 80,
      "ckpt_every": 72,
      "keep_every": 72,
    
      "name": model_name,
      "wandb_project": "mesh-transformer-jax",
      "comment": ""
    }
    
    
    
    # def parse_args():
    #     # Parse command line arguments
    #     parser = argparse.ArgumentParser(description="""
    #     To use, download the full checkpoint archive, extract and upload to a GCS bucket, and set that as --tune-model-path
    #     Modify the config file:
    #         - set `model_dir` to where the checkpoints should be written during training
    #         - set `train_set`, `val_set` to index files for your data
    #         - set `tpu_size` to 8 (if on a v3-8)
    #         - set `warmup_steps`, `anneal_steps`, `lr`, `end_lr` to the lr schedule for your finetuning run
    #         - the global step will reset to 0, keep that in mind when writing your lr schedule
    #         - set `name` to specify the name of the Weights & Biases run
    #         - set `wandb_project` to specify the Weights & Biases project to log to
    #     To prepare data in the expected data format:
    #         - use the script `create_finetune_tfrecords.py` in this repo to create data in the expected format
    #         - upload the .tfrecords files to GCS
    #         - save their GCS paths to a index file under `data/`, see existing files for examples
    #     """,
    #     formatter_class=argparse.RawTextHelpFormatter)
    #     # parser.add_argument("--config", type=str, default=None, help="Config file location")
    #     parser.add_argument("--tune-model-path", type=str, default = 'gs://nlp-project0/step_383500/step_383500/', help="Base model to finetune")
    #     parser.add_argument("--fresh-opt", default=False, action="store_true", help="Use a newly initialized optimizer, ignoring any optimizer state saved in the base checkpoint")
    
    #     args = parser.parse_args()
    #     return args
    
    
    # def save(network, step, bucket, path, mp, aux=None, keep_n=3, delete_old=True):
    #     assert path
    #     client = storage.Client()
    
    #     if aux is None:
    #         aux = {}
    
    #     try:
    #         with open(f"gs://{bucket}/{path}/meta.json", "r") as f:
    #             meta = json.load(f)
    #     except:
    #         # create metadata file
    #         with open(f"gs://{bucket}/{path}/meta.json", "w") as f:
    #             json.dump({
    #                 "step": 0,
    #                 "checkpoints": [],
    #                 "aux": {}
    #             }, f)
    
    #     # do sharded checkpoint writing
    #     start = time.time()
    #     res = []
    #     for shard_id in range(mp):
    #         write_ckpt(network.state, f"gs://{bucket}/{path}/step_{step}/", shard_id)
    
    #     print(f"Wrote checkpoint in {time.time() - start:.06}s")
    
    #     with open(f"gs://{bucket}/{path}/meta.json", "r") as f:
    #         meta = json.load(f)
    
    #     meta["step"] = step
    #     meta["checkpoints"].append(step)
    #     all_aux = meta.get("aux", {})
    
    #     while len(meta["checkpoints"]) > keep_n:
    #         ckpt_to_delete = meta["checkpoints"].pop(0)
    
    #         try:
    #             del all_aux[str(ckpt_to_delete)]
    #         except:
    #             print(f"failed to delete the aux state for {step}")
    
    #         if delete_old:
    #             print(f"deleting checkpoint {ckpt_to_delete}")
    #             for blob in client.list_blobs(bucket, prefix=f"{path}/step_{ckpt_to_delete}/"):
    #                 # print(f"deleting {blob.name}")
    #                 assert path in blob.name
    #                 blob.delete()
    #         else:
    #             print(f"keeping checkpoint {ckpt_to_delete}")
    
    #     all_aux[step] = aux
    #     meta["aux"] = all_aux
    
    #     with open(f"gs://{bucket}/{path}/meta.json", "w") as f:
    #         json.dump(meta, f)
    
    
    # def train_step(network, data):
    #     inputs = {
    #         "obs": data[:, :, :-1],
    #         "target": data[:, :, 1:],
    #     }
    
    #     loss, last_loss, grad_norm, grad_norm_micro = network.train(inputs)
    
    #     return (
    #         np.array(loss).mean(),
    #         np.array(last_loss).mean(),
    #         np.array(grad_norm).mean(),
    #         np.array(grad_norm_micro).mean(),
    #     )
    
    
    # def eval_step(network, data):
    #     inputs = {
    #         "obs": data[:, :-1],
    #         "target": data[:, 1:],
    #     }
    
    #     out = network.eval(inputs)
    #     loss = out["loss"]
    
    #     return np.array(loss).mean()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # if __name__ == "__main__":
    #     args = parse_args()
    #     # params = json.load(open(args.config))
    
    #     gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    #     per_replica_batch = params["per_replica_batch"]
    #     cores_per_replica = params["cores_per_replica"]
    
    #     assert cores_per_replica <= 8
    
    #     bucket = params["bucket"]
    #     model_dir = params["model_dir"]
    #     layers = params["layers"]
    #     d_model = params["d_model"]
    #     n_heads = params["n_heads"]
    #     n_vocab = params["n_vocab"]
    #     seq = params["seq"]
    #     norm = params["norm"]
    
    #     val_batches = params["val_batches"]
    #     val_every = params["val_every"]
    #     ckpt_every = params["ckpt_every"]
    #     keep_every = params["keep_every"]
    #     # eval_tasks = params["eval_harness_tasks"]
    #     total_steps = params["total_steps"]
    
    #     pe = params["pe"]
    #     assert pe in ["fixed", "rotary", "t5"]
    
    #     warmup_steps = params["warmup_steps"]
    #     anneal_steps = params["anneal_steps"]
    #     lr = params["lr"]
    #     end_lr = params["end_lr"]
    #     weight_decay = params["weight_decay"]
       
    #     # alpha parameter for the exponential moving averages used to compute B_simple
    #     noise_scale_alpha = params.get("noise_scale_alpha", 0.01)
    
    #     scheduler = util.gpt3_schedule(warmup_steps, anneal_steps, lr, end_lr)
        
    #     opt = optax.chain(
    #         optax.scale(1 / gradient_accumulation_steps),
    #         clip_by_global_norm(1),
    #         optax.scale_by_adam(),
    #         additive_weight_decay(weight_decay),
    #         optax.scale(-1),
    #         optax.scale_by_schedule(scheduler)
    #     )
    
    #     params["optimizer"] = opt
    
    #     start = time.time()
    #     tpu_size = jax.device_count()
    #     if tpu_size < cores_per_replica:
    #         msg = f"each shard needs a separate device, but device count ({tpu_size}) < shard count ({cores_per_replica})"
    #         raise ValueError(msg)
    #     print(f"jax devices: {tpu_size}")
    #     print(f"jax runtime initialized in {time.time() - start:.06}s")
    
    #     mesh_shape = (tpu_size // cores_per_replica, cores_per_replica)
    #     devices = np.array(jax.devices()).reshape(mesh_shape)
    
    #     # pick initial ckpt - based on tuning vs train from scratch
    
    #     step = 0
    #     initial_ckpt_state_path = None
    #     train_loader = None
    
    #     if args.tune_model_path:
    #         print('`--tune_model_path` passed: we are beginning a fine-tuning run')
    #         fine_tuning = True
    #         initial_ckpt_state_path = args.tune_model_path
    #     else:
    #         print('`--tune_model_path` not passed: we are continuing a fine-tuning run from a checkpoint (or we are not fine-tuning)')
    #         fine_tuning = False
    #         initial_ckpt_model_dir = model_dir
    #         initial_ckpt_path = f"gs://{bucket}/{initial_ckpt_model_dir}"
    #         meta_path = f"{initial_ckpt_path}/meta.json"
    
    #         try:
    #             with open(meta_path, "r") as f:
    #                 meta = json.load(f)
    #             ckpt_step = meta["checkpoints"][-1]
    #             initial_ckpt_state_path = f"{initial_ckpt_path}/step_{ckpt_step}/"
    #             print(f"state will be restored from checkpoint {ckpt_step}")
    
    #             step = ckpt_step
    #             train_loader = meta['aux'][str(ckpt_step)].get("train_loader", None)
    #         except NotFound:
    #             # no checkpoint, start at zero
    #             print(f"No checkpoint to load at {initial_ckpt_path}. Training from scratch.")
    
    #     if initial_ckpt_state_path:
    #         print(f"path to load checkpoint from: {initial_ckpt_state_path}")
    #     else:
    #         print("not loading from a checkpoint")
    
    #     # set up datasets
    #     print("setting up datasets")
    
    #     train_dataset = TFRecordNewInputs(f"{params['train_set']}",
    #                                       batch_size=(
    #                                       gradient_accumulation_steps,
    #                                       per_replica_batch * tpu_size // cores_per_replica),
    #                                       sample_size=params['seq'],
    #                                       restore_state=train_loader)
    
    
    #     global_val_batch = per_replica_batch * tpu_size // cores_per_replica
    
    #     val_sets = {}
    
    #     for k, v in params["val_set"].items():
    #         val_sets[k] = TFRecordNewInputs(
    #             f"data/{v}", batch_size=(global_val_batch,), sample_size=seq
    #         )
    
    #     # tok/sec metrics
    #     sequences_per_step = gradient_accumulation_steps * (per_replica_batch * tpu_size // cores_per_replica)
    #     tokens_per_step = params['seq'] * sequences_per_step
    
    #     # load + run
    #     with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
    #         print("initializing network")
    #         network = CausalTransformer(params)
    
    #         if initial_ckpt_state_path:
    #             print("loading network")
    #             if fine_tuning:
    #                 # get the scheduler step stored in the just-initialized optimizer
    #                 # should be zero
    #                 init_sched_state = network.state["opt_state"][-1]
    
    #             start = time.time()
    #             network.state = read_ckpt(network.state, initial_ckpt_state_path, devices.shape[1], load_opt=(not args.fresh_opt))
    
    #             if fine_tuning:
    #                 # overwrite the loaded scheduler step with zeros
    #                 # this makes fine-tuning use the lr schedule in
    #                 network.state["opt_state"][-1] = init_sched_state
    
    #             print(f"network loaded in {time.time() - start:.06}s")
    
    #         print('compiling train fn')
    #         start = time.time()
    #         loss, last_loss, grad_norm, grad_norm_micro = train_step(
    #             network, train_dataset.get_samples()
    #         )
    #         step += 1
    #         print(f"Train fn compiled in {time.time() - start:.06}s")
    
    #         print('compiling eval fn')
    #         start = time.time()
    #         for val_set in val_sets.values():
    #             eval_step(network, val_set.get_samples())
    #             val_set.reset()
    #         print(f"Eval fn compiled in {time.time() - start:.06}s")
    
    #         # project = params.get("wandb_project", "mesh-transformer-jax")
    #         # wandb.init(project=project, name=params["name"], config=params)
    
    #         G_noise_avg = None
    #         S_noise_avg = None
    #         print('step1: ' + str(step))
    #         while True:
    #             if (step % ckpt_every == 1) or step == total_steps:
    #                 print(f"saving a checkpoint for step {step}")
    #                 save(network, step, bucket, model_dir,
    #                      mp=cores_per_replica,
    #                      aux={"train_loader": train_dataset.get_state()},
    #                      delete_old=True,
    #                      )
    
    #             if step % val_every == 1:  # 1 because we've already taken a step to compile train fn
    #                 for name, val_set in val_sets.items():
    #                     val_loss = []
    #                     for i, _ in tqdm(zip(val_set.sample_once(), range(val_batches)),
    #                                      desc=f"validation for step {step}, set {name}",
    #                                      total=val_batches):
    #                         val_loss.append(eval_step(network, i))
    #                     val_set.reset()
    
    #                     val_loss = np.array(val_loss).mean()
    #                     print(f"validation loss for step {step}, set {name}: {val_loss}")
    
    #                     # wandb.log({f'val/loss_{name}': float(val_loss)}, step)
    #             print('step2: ' + str(step))
    #             if step == total_steps:
    #                 print("training completed!")
    #                 # exit()
    #                 break
                
    #             # continue
              
    #             # start = time.time()
    #             # loss, last_loss, grad_norm, grad_norm_micro = train_step(
    #             #     network, train_dataset.get_samples()
    #             # )
    #             step += 1
                
                
    #         del network

                
    # #### -------------------------------- Slim model -----------------------------------

    # def parse_args_2():
    #     # Parse command line arguments
    #     parser = argparse.ArgumentParser()
    #     # parser.add_argument("--config", type=str, default=None, help="Config file location")
    #     parser.add_argument("--ckpt-step", type=int, default=-1, help="Step number of the checkpoint to convert (if not specified, converts the most recent checkpoint)")
    #     parser.add_argument("--f16", default=False, action="store_true", help="Convert to float16 (instead of bfloat16)")
    
    #     args = parser.parse_args()
    #     return args

    # if __name__ == "__main__":
    #     args = parse_args_2()
    #     # params = json.load(open(args.config))
    #     convert_fn = to_f16 if args.f16 else to_bf16
    
    #     cores_per_replica = params["cores_per_replica"]
    
    #     assert cores_per_replica <= 8
    
    #     bucket = params["bucket"]
    #     model_dir = params["model_dir"]
    
    #     params["optimizer"] = optax.chain(
    #         optax.scale(1),
    #         clip_by_global_norm(1),
    #         optax.scale_by_adam(),
    #         optax.additive_weight_decay(0),
    #         optax.scale(-1),
    #         optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
    #     )
    
    #     start = time.time()
    #     print(f"jax devices: {jax.device_count()}")
    #     print(f"jax runtime initialized in {time.time() - start:.06}s")
    
    #     mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    #     devices = np.array(jax.devices()).reshape(mesh_shape)
    
    #     with open(f"gs://{bucket}/{model_dir}/meta.json", "r") as f:
    #         meta = json.load(f)
    
    #     if args.ckpt_step > -1:
    #         ckpt_step = args.ckpt_step
    #     else:
    #         ckpt_step = meta["checkpoints"][-1]
    #     print(f"using checkpoint {ckpt_step}")
    
    #     with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
    #         network = CausalTransformer(params)
    
    #         start = time.time()
    #         network.state = read_ckpt(network.state, f"gs://{bucket}/{model_dir}/step_{ckpt_step}/", devices.shape[1])
    #         print(f"network loaded in {time.time() - start:.06}s")
    
    #         start = time.time()
    #         del network.state["opt_state"]
    
    #         network.state["params"] = convert_fn(network.state["params"])
    #         print(f"network converted in {time.time() - start:.06}s")
    
    #         suffix = "_slim_f16" if args.f16 else "_slim"
    
    #         for i in range(cores_per_replica):
    #             write_ckpt(network.state, f"gs://{bucket}/{model_dir}{suffix}/step_{ckpt_step}/", i)
    #             print(f"written shard {i}")


    # del network
    
    
    
    # ## Cleaning cloud (1)

    # storage_client_1 = storage.Client()
    # bucket_1 = storage_client_1.get_bucket('nlp-project0')
    # blobs_1 = bucket_1.list_blobs(prefix = "finetuned_models_stage_1/" + ticker + '/' + str(year) + '/')
    
    # for blob in blobs_1:
    #   blob.delete()
    
    ####
    # python to_hf_weights.py --input-ckpt ./step_383500 --config ./configs/6B_roto_256.json --output-path ./gpt-j-6B --cpu --dtype fp32
    ####
    
    
 
    input_check_point_path = 'gs://nlp-project0/finetuned_models_stage_1/' + ticker + '/' + str(year) + '_slim/step_72/'
    output_check_point_path = 'gs://nlp-project0/finetuned_models/' + ticker + '/' + str(year) + '/'
    
    # xla: tell jax to not pre allocate all device memory
    # and only allocate memory as needed.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    
    DEBUG = False
    
    parser = argparse.ArgumentParser(
        description=(
            "Used to turn a sharded trained gpt-j checkpoint into pytorch hugging face format."
            "This script works best on a slimmed checkpoint (full checkpoints can be used but require ~100gb of ram)."
            "Currently, weights must be split into 8 shards for this to work."
            "All paths can be local or google cloud storage paths. S3 paths supported as well with `pip install pathy[s3]`."
            "Can be run on tpu, or on gpu with `pip install --upgrade jax==0.2.12 jaxlib==0.1.67+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html`"
        )
    )
    parser.add_argument(
        "--input-ckpt",
        type=str,
        required=False,
        help='path to model checkpoint folder. Google storage can be used with "gs://bucket/path/step_{n}" format.',
        default = input_check_point_path,
        metavar="path",
    )
    parser.add_argument(
        "--config", type=str, required=False, help="Config file location", metavar="path"
    )
    parser.add_argument(
        "--output-path",
        required=False,
        type=str,
        help='Full path to save checkpoint to. Google storage can be used with "gs://bucket/path" format.',
        default = output_check_point_path
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose printing.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default = True,
        help="Run resharding on cpu instead of searching for jax device (i.e. gpu/tpu). Will default to cpu if jax wasn't installed with `+cuda110` option",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        help="One of fp32, fp16 or bf16. Default=fp16. WARNING: Experimental. Make sure to check weights after conversion to make sure dtype information is retained.",
    )
    
    
    def process_args_3(
        input_ckpt: Union[FluidPath, str],
        config: Union[FluidPath, str],
        output_path: Union[FluidPath, str],
        dtype: str = "fp16",
        cpu: bool = False,
        **kwargs,
    ):
        # validate paths and turn them into Pathy paths.
        input_ckpt = Pathy.fluid(str(input_ckpt))
        assert input_ckpt.is_dir(), f'no such directory "{input_ckpt}"'
        config = Pathy.fluid(str(config))
        # assert config.is_file(), f'no such file "{config}"'
        first_shard = input_ckpt / "shard_0"
        assert first_shard.is_dir(), f'no shards found at "{input_ckpt}"'
    
        output_path = Pathy.fluid(str(output_path))
        output_path.mkdir(exist_ok=True)
    
        # make sure dtype is valid
        assert dtype in {"fp16", "fp32", "bf16"}
        np_dtype = np.float16
        torch_dtype = torch.float16
        if dtype != "fp16":
            warnings.warn(
                "WARNING: Dtype support other than fp16 is Experimental. Make sure to check weights after conversion to make sure dtype information is retained."
            )
            if dtype == "bf16":
                # np doesn't have bfloat16 so float32 is used to retain information before converting to torch.
                np_dtype = np.float32
                torch_dtype = torch.bfloat16
            elif dtype == "fp32":
                np_dtype = np.float32
                torch_dtype = torch.float32
    
        # tell jax to run on cpu instead of gpu/tpu
        if cpu:
            jax.config.update("jax_platform_name", "cpu")
    
        return input_ckpt, config, output_path, np_dtype, torch_dtype
    
    
    def tree_flatten_with_names(pytree, is_leaf, path="", to_id=id):
        id_to_name = {}
        if getattr(pytree, "items", None):
            for k, v in pytree.items():
                k_path = f"{path}/{k}"
                if is_leaf(v):
                    id_to_name[to_id(v)] = k_path
                else:
                    id_to_name = {
                        **id_to_name,
                        **tree_flatten_with_names(v, is_leaf=is_leaf, path=k_path),
                    }
        elif getattr(pytree, "__getitem__", None):
            for v in pytree:
                if is_leaf(v):
                    id_to_name[to_id(v)] = path
                else:
                    id_to_name = {
                        **id_to_name,
                        **tree_flatten_with_names(v, is_leaf=is_leaf, path=path),
                    }
        else:
            id_to_name[to_id(pytree)] = path
        return id_to_name
    
    
    def tree_leaves_with_names(pytree, to_id=id):
        leaves = jax.tree_leaves(pytree)
        is_leaf = lambda x: not isinstance(x, list) and to_id(x) in [
            to_id(x) for x in leaves
        ]
        return tree_flatten_with_names(pytree, is_leaf)
    
    
    def get_tree_leaves_names_reduced(pytree) -> List[str]:
    
        leaves_ids = tree_leaves_with_names(pytree, to_id=id)
        leaves = jax.tree_leaves(pytree)
        return [leaves_ids[id(l)] for l in leaves]
    
    
    layer_2_hf_inner_module_id = {
        "linear": "attn.q_proj",
        "linear_1": "attn.v_proj",
        "linear_2": "attn.k_proj",
        "linear_3": "attn.out_proj",
        "linear_4": "mlp.fc_in",
        "linear_5": "mlp.fc_out",
        "replicated_layer_norm": "ln_1",
    }
    
    projection_layer_2_hf_id_start = {
        "linear": "lm_head",
        "replicated_layer_norm": "transformer.ln_f",
    }
    
    
    def leave_name_to_hf_layer_id(leaf_name: str):
        if not leaf_name.startswith("/params"):
            if leaf_name == "/step":
                return None
            else:
                raise NotImplementedError(f"Unknown leaf name: {leaf_name}")
    
        match = re.search(
            r"\/params\/causal_transformer_shard\/~\/(?P<module_name>.*)\/~\/(?P<layer_name>.*)\/(?P<wb>.*)",
            leaf_name,
        )
    
        assert match, f'couldn\'t match pattern against: "{leaf_name}"'
    
        layer_name = match["layer_name"]
        module_name = match["module_name"]
        wb = match["wb"]
    
        if wb in {"w", "scale"}:
            weight_or_bias = "weight"
        elif wb in {"b", "offset"}:
            weight_or_bias = "bias"
        else:
            raise NotImplementedError(
                f"unknown weight/bais type identifier \"{wb}\" at end of: '{leaf_name}'"
            )
    
        # switch statement based on top level module name
        if module_name == "embedding_shard":
            hf_id = f"transformer.wte.{weight_or_bias}"
    
        elif module_name.startswith("layer"):
            module_index = int(module_name.split("_")[-1])
            hf_inner_module_id = layer_2_hf_inner_module_id[layer_name]
            hf_id = f"transformer.h.{module_index}.{hf_inner_module_id}.{weight_or_bias}"
    
        elif module_name == "projection_shard":
            hf_id = f"{projection_layer_2_hf_id_start[layer_name]}.{weight_or_bias}"
    
        else:
            raise NotImplementedError(
                f"unknown leaf module type \"{module_name}\" in: '{leaf_name}'"
            )
    
        if DEBUG:
            print(f"{leaf_name} \n\t -> {hf_id}")
    
        return hf_id
    
    
    # TODO(nijkamp): rewrite this mess
    def reshard(x, old_shape, do_shard_ln, do_shard_bias):
        # reshards using numpy arrays so as to not fill up jax memory
        if len(x.shape) == 1:
            out = np.array(x[0:1])
    
        elif len(x.shape) == 2:
            if do_shard_ln:
                out = np.array(x[0:1])
            elif do_shard_bias:
                out = np.reshape(np.sum(x, axis=0), old_shape)
            else:
                out = x.reshape(old_shape)
    
        elif len(x.shape) == 3:
            if x.shape[0] * x.shape[2] == old_shape[2]:
                out = np.transpose(x, (1, 0, 2)).reshape(old_shape)
            elif x.shape[0] * x.shape[1] == old_shape[1]:
                out = np.reshape(x, old_shape)
            else:
                raise NotImplementedError(f"unimplemented, {x.shape}, {old_shape}")
        else:
            raise NotImplementedError(f"unimplemented, {x}")
        return out
    
    
    def read_npz(fpath: FluidPath):
        # read npz file of ndarrays
        with fpath.open("rb") as f:
            buf = f.read()
            f_io = io.BytesIO(buf)
            deserialized = np.load(
                f_io,
            )
            assert isinstance(
                deserialized, np.lib.npyio.NpzFile
            ), f"Not an npz file {type(deserialized)=} {f=}"
            # arrays are only loaded when accessed. So we need to access them before returning
            arrays = []
            for i in deserialized:
                arr = deserialized[i]
                assert isinstance(arr, np.ndarray), f"Not a np.ndarray {type(arr)=} {f=}"
                arrays.append(arr)
            return arrays
    
    
    def read_file_shards(
        ckpt_dir: FluidPath, fname: str, shards_in: int
    ) -> List[List[np.ndarray]]:
        # read same file like "12.npz" across all shard directories
        with multiprocessing.pool.ThreadPool(shards_in) as p:
            return list(
                p.imap(
                    read_npz,
                    [ckpt_dir / f"shard_{i}" / fname for i in range(shards_in)],
                )
            )
    
    
    def lazy_read_ckpt_shards(
        ckpt_dir: FluidPath, shards_in: int, pieces: int = 16, reverse: bool = True
    ):
        for i in range(pieces):
            # iterate through files in direction of choice
            fname = f"{(pieces-1) - i}.npz" if reverse else f"{i}.npz"
            if DEBUG:
                print(f"reading from {fname}")
            file_shards = read_file_shards(ckpt_dir, fname, shards_in)
    
            # iterate over layers in file returning all shards for each
            file_shards = list(zip(*file_shards))
            if reverse:
                file_shards = reversed(file_shards)
            yield from file_shards
    
    
    def unshard_leave(
        leave_shards: Iterable[np.ndarray],
        leave_name: str,
        old_shape: List[int],
        np_dtype=np.float16,
    ):
        # reshard all leave shards into single shard.
    
        # stack leave shards into single np.ndarray
        x = np.stack(leave_shards)
        # assert isinstance(x, jnp.ndarray)
    
        # As far as i can tell, this just re labels the dtype of arrays
        # labeled with "V2" dtype. In theory, V2 was just an alias for bfloat16
        # which needs to be relabeled in order for it to be understood.
        if x.dtype == np.dtype("V2"):
            x.dtype = jnp.bfloat16
    
        if DEBUG:
            print(f"RESHARDING: {leave_name=} {x.shape=} {old_shape=}")  # type: ignore
    
        # transform sharded array to match old_shape
        x = reshard(
            x,
            old_shape,
            do_shard_bias=leave_name.endswith("embedding_shard/~/linear/b")
            or leave_name.endswith("linear_5/b"),
            do_shard_ln=leave_name.endswith("replicated_layer_norm/offset")
            or leave_name.endswith("replicated_layer_norm/scale"),
        )
        assert (
            x.shape == old_shape
        ), f"Incompatible checkpoints {x.shape} vs {old_shape} {leave_name}"
        return x.astype(np_dtype)
    
    
    def save_pytree_as_hf(
        pytree,
        input_ckpt: FluidPath,
        shards_in: int,
        output_path: FluidPath,
        n_layers: int = 28,
        np_dtype: type = np.float16,
        torch_dtype: torch.dtype = torch.float16,
        n_seq: int = 2048,
    ):
        # Loads layers and names in reverse order to avoid loading unneeded opt_state layers
        # that are at the front of full (i.e. not slim) models.
    
        old_leave_shapes = [old.shape for old in jax.tree_flatten(pytree)[0]]
        leave_names = get_tree_leaves_names_reduced(pytree)
        del pytree
    
        assert len(old_leave_shapes) == len(
            leave_names
        ), f"{len(old_leave_shapes)=}  {len(leave_names)=}"
        # get generator that emits all shards of leaves from npz files in reverse order
        loaded_shards_in = lazy_read_ckpt_shards(input_ckpt, shards_in, reverse=True)
    
        print("Reading and transforming layers/shards. This may take a while.")
    
        hf_checkpoint = {}
        wte_first = None  # saves first instance of a wte weight in order to combine it with the second.
        # Reverse iteration to grab leave_names and old leaves from the back
        for i in tqdm(
            reversed(range(len(leave_names))),
            desc="Reading/Transforming Layers",
            total=len(leave_names),
        ):
    
            # load next shard with correstponding leave name and old shape
            x = next(loaded_shards_in)
            leave_name = leave_names[i]
            old_shape = old_leave_shapes[i]
            hf_layer_id = leave_name_to_hf_layer_id(leave_name)
    
            # If leave is not needed in hf model (/step')
            if not hf_layer_id:
                continue
    
            x = unshard_leave(x, leave_name, old_shape, np_dtype=np_dtype)
            # remove first empty dimension and transpose.
            x = torch.tensor(x.squeeze(0), dtype=torch_dtype).T
    
            # wte embedding weights/bias need to be combined since hf model has no wte.embedding.bias
            if hf_layer_id.startswith("transformer.wte"):
                # un/re-transpose since wte weight is only leave that shouldn't be transposed
                x = x.T
                # store first weight/bias then skip saving
                if wte_first is None:
                    wte_first = x
                    continue
                # combine second wte bias/weight with first then move on to saving with weight name
                else:
                    x = x + wte_first
                    hf_layer_id = "transformer.wte.weight"
    
            # save params as single file with proper hf id mapped to them in save_map
            hf_checkpoint[hf_layer_id] = x
    
        # add attention bias layers
        attn_bias_weights = torch.tril(torch.ones((n_seq, n_seq), dtype=torch.bool)).view(
            1, 1, n_seq, n_seq
        )
        attn_masked_bias_weights = torch.tensor(-1e9, dtype=torch_dtype)
    
        for i in range(n_layers):
            hf_checkpoint[f"transformer.h.{i}.attn.bias"] = attn_bias_weights
            hf_checkpoint[f"transformer.h.{i}.attn.masked_bias"] = attn_masked_bias_weights
    
        torch.save(hf_checkpoint, (output_path / "pytorch_model.bin").open(mode="wb"))
    
    
    def save_config_to_hf_format(params: dict, torch_dtype: str, output_path: FluidPath):
    
        config = {
            "activation_function": "gelu_new",
            "architectures": ["GPTJForCausalLM"],
            "attn_pdrop": 0.0,
            "bos_token_id": 50256,
            "embd_pdrop": 0.0,
            "eos_token_id": 50256,
            "gradient_checkpointing": False,
            "initializer_range": 0.02,
            "layer_norm_epsilon": 1e-05,
            "model_type": "gptj",
            "n_embd": params["d_model"],
            "n_head": params["n_heads"],
            "n_layer": params["layers"],
            "n_positions": params["seq"],
            "rotary_dim": params["pe_rotary_dims"],
            "summary_activation": None,
            "summary_first_dropout": 0.1,
            "summary_proj_to_labels": True,
            "summary_type": "cls_index",
            "summary_use_proj": True,
            "transformers_version": "4.10.0.dev0",
            "tokenizer_class": "GPT2Tokenizer",
            "task_specific_params": {
                "text-generation": {"do_sample": True, "temperature": 1.0, "max_length": 50}
            },
            "torch_dtype": str(torch_dtype).split(".")[-1],
            "use_cache": True,
            "vocab_size": params["n_vocab"],
        }
    
        with (output_path / "config.json").open("w") as f:
            json.dump(config, f, indent=2)
    
    
    def save_sharded_to_hf_format(
        input_ckpt: Union[FluidPath, str],
        params: dict,
        output_path: Union[FluidPath, str],
        cpu: bool = False,
        dtype: str = "fp16",
    ):
    
        devices = np.array([jax.devices()[0]]).reshape((1, 1))
        with maps.mesh(devices, ("dp", "mp")):
            params_local = params.copy()
            params_local["cores_per_replica"] = maps.thread_resources.env.shape["mp"]
            network = CausalTransformer(params_local)
    
            save_pytree_as_hf(
                network.state,
                input_ckpt=input_ckpt,
                shards_in=params["cores_per_replica"],
                output_path=output_path,
                n_layers=params["layers"],
                np_dtype=np_dtype,
                torch_dtype=torch_dtype,
                n_seq=params["seq"],
            )
    
    
    if __name__ == "__main__":
        args = vars(parser.parse_args())
    
        DEBUG = args["debug"]
        start = time.time()
    
        input_ckpt, config, output_path, np_dtype, torch_dtype = process_args_3(**args)
        # params = json.load(open(config))
        params["optimizer"] = optax.scale(0)
    
        save_sharded_to_hf_format(input_ckpt, params, output_path, np_dtype, torch_dtype)
        save_config_to_hf_format(params, torch_dtype, output_path)
        print(
            f"HF weights created in {(time.time() - start):.0f}s \"{args['output_path']}\""
        )
    
    
    ## Cleaning cloud (2)

    storage_client_2 = storage.Client()
    bucket_2 = storage_client_2.get_bucket('nlp-project0')
    blobs_2 = bucket_2.list_blobs(prefix = 'finetuned_models_stage_1/' + ticker + '/' + str(year) + '_slim/')
    
    for blob in blobs_2:
      blob.delete()












